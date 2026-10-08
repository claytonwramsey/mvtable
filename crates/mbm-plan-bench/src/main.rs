//! End-to-end motion-planning benchmark.
//!
//! Output goes to `data/mbm_plan_results.csv` at the workspace root.

use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use capt::Capt;
use carom::{
    BlockValidate, Robot,
    env::World3d,
    robot::{Baxter, Fetch, Panda, Ur5},
};
use kiddo::ImmutableKdTree;
use mbm::{Problem, dir_to_problems};
use mbm_plan_bench::{
    PointCloudWorld, SimdPointCloudWorld, is_sampleable, sample_scene, solve_with_backend,
};
use mvt_cpp::MvtCpp;
use mvtable::{MutableMvt, Mvt};
use mvtable_bench::{
    filter::centervox_filter,
    timing::{round_robin_medians, time_build},
};
use nalgebra::{Isometry3, Vector3};
use rand::{SeedableRng, rngs::SmallRng};

/// The 7 canonical MotionBenchMaker benchmark environments, used by every robot except Baxter
/// (see [`BAXTER_DATASETS`]).
const DATASETS: [&str; 7] = [
    "bookshelf_small",
    "bookshelf_tall",
    "bookshelf_thin",
    "box",
    "cage",
    "table_pick",
    "table_under_pick",
];

/// Baxter's MotionBenchMaker problem set only covers one environment, at three difficulties, with
/// both arms.
const BAXTER_DATASETS: [&str; 3] = [
    "bookshelf_tall_both_arms_easy",
    "bookshelf_tall_both_arms_hard",
    "bookshelf_tall_both_arms_medium",
];

/// Maximum number of scenes to solve (per backend) per dataset. Overridable via the
/// `MBM_PLAN_BENCH_MAX_SCENES` environment variable for a quick dev-loop run.
const MAX_PROBLEMS_PER_DATASET: usize = 50;

/// Surface-sample density (points per unit area).
const DENSITY: f32 = 6000.0;

/// The fixed point-cloud filter resolution used for every problem, as a multiple of each robot's
/// own smallest collision-sphere radius (`Robot::MIN_RADIUS`).
const R_FILTER_SCALE: f32 = 4.0;

/// Per-robot `mvtable::Mvt`/`MutableMvt` voxel width, tuned by `mbm_bench`'s per-robot
/// hyperparameter sweep (SIMD `collides_simd` throughput, lanes=8).
const PANDA_VOXEL_WIDTH: f32 = 0.14;
const UR5_VOXEL_WIDTH: f32 = 0.19;
const FETCH_VOXEL_WIDTH: f32 = 0.17;
const BAXTER_VOXEL_WIDTH: f32 = 0.14;

/// Wall-clock cutoff per (backend, problem) solve attempt, so a pathologically slow combination
/// can't stall the whole unattended run.
const MAX_SOLVE_TIME: Duration = Duration::from_secs(10);

/// Median construction time of every backend on each scene of one dataset.
///
/// Times are measured with [`round_robin_medians`], so no backend ever builds the same scene twice
/// in a row.
/// Entry `i` of each field corresponds to scene `i`, and is `None` if that backend cannot be built
/// for that scene.
/// The SIMD and scalar variants of a backend build the same structure, so they share a field,
/// except for `capt`, whose lane count changes what it builds.
struct BuildTimes {
    primitive: Vec<Option<Duration>>,
    mvtable: Vec<Option<Duration>>,
    mvtable_mutable: Vec<Option<Duration>>,
    mvt_cpp: Vec<Option<Duration>>,
    capt: Vec<Option<Duration>>,
    capt_simd: Vec<Option<Duration>>,
    kiddo: Vec<Option<Duration>>,
}

impl BuildTimes {
    fn measure<const N: usize>(
        scenes: &[(&Problem<N>, usize, Vec<[f32; 3]>)],
        voxel_width: f32,
        r_range: (f32, f32),
        mvt_cpp_r_range: (f32, f32),
    ) -> Self {
        let n = scenes.len();
        let points = |i: usize| scenes[i].2.as_slice();
        Self {
            primitive: round_robin_medians(n, |i| Some(time_build(|| scenes[i].0.world.clone()).0)),
            mvtable: round_robin_medians(n, |i| {
                Some(time_build(|| Mvt::<3, f32>::new(points(i), voxel_width)).0)
            }),
            mvtable_mutable: round_robin_medians(n, |i| {
                Some(time_build(|| MutableMvt::<3, f32>::new(points(i), voxel_width)).0)
            }),
            mvt_cpp: round_robin_medians(n, |i| {
                let (t, built) = time_build(|| MvtCpp::try_new(points(i), mvt_cpp_r_range));
                built.is_ok().then_some(t)
            }),
            capt: round_robin_medians(n, |i| {
                Some(time_build(|| Capt::<3, f32, u32>::new(points(i), r_range, 1)).0)
            }),
            capt_simd: round_robin_medians(n, |i| {
                Some(time_build(|| Capt::<3, f32, u32>::new(points(i), r_range, 8)).0)
            }),
            kiddo: round_robin_medians(n, |i| {
                let (t, built) =
                    time_build(|| ImmutableKdTree::<f32, 3>::new_from_slice(points(i)));
                built.is_ok().then_some(t)
            }),
        }
    }
}

fn max_problems_per_dataset() -> usize {
    std::env::var("MBM_PLAN_BENCH_MAX_SCENES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(MAX_PROBLEMS_PER_DATASET)
}

/// Build `structure_name`'s backend from `filtered_points`, run the planner against it, and write
/// one CSV row with the outcome and `construction`, the backend's median construction time from
/// [`BuildTimes`].
/// Logs and returns `Ok(())` without writing a row if this backend/problem combination fails to
/// construct or solve.
#[expect(
    clippy::too_many_arguments,
    reason = "internal driver, not a public API"
)]
fn run_one_backend<R, W, const N: usize>(
    robot: &R,
    robot_name: &str,
    dataset: &str,
    problem: &Problem<N>,
    filtered_points: &[[f32; 3]],
    r_range: (f32, f32),
    r_filter: f32,
    name: &str,
    construction: Option<Duration>,
    // An `Err` means "skip this backend for this problem, don't write a row" - in practice only
    // `MvtCpp::try_new`
    builder: impl Fn(&[[f32; 3]], (f32, f32)) -> Result<W, Box<dyn std::error::Error>>,
    csv: &mut impl Write,
) -> Result<(), Box<dyn std::error::Error>>
where
    R: Robot<N, f32> + BlockValidate<N, f32, W> + Clone,
{
    let (Ok(structure), Some(construction)) = (builder(filtered_points, r_range), construction)
    else {
        eprintln!("  {name}: skipping {robot_name}/{dataset}#{}", problem.id);
        return Ok(());
    };
    let construction_secs = construction.as_secs_f64();

    let result = match solve_with_backend(
        robot.clone(),
        robot_name,
        problem,
        structure,
        MAX_SOLVE_TIME,
    ) {
        Ok(result) => result,
        Err(e) => {
            eprintln!(
                "  {name}: skipping {robot_name}/{dataset}#{}: {e}",
                problem.id
            );
            return Ok(());
        }
    };
    let solved = result.trajectory.is_some();
    println!(
        "  {name}: {robot_name}/{dataset}#{} solved={solved} {:.3}s (+{:.3}s construction, {} \
         samples, {} nodes)",
        problem.id,
        result.time.as_secs_f64(),
        construction_secs,
        result.samples,
        result.nodes,
    );
    writeln!(
        csv,
        "{name},{robot_name},{dataset},{},{r_filter},{},{solved},{},{construction_secs},{},{}",
        problem.id,
        filtered_points.len(),
        result.time.as_secs_f64(),
        result.samples,
        result.nodes,
    )?;
    csv.flush()?;
    Ok(())
}

/// Solve up to [`max_problems_per_dataset`] scenes from every dataset in `datasets`, for a single
/// robot, once per collision-checking backend, appending a CSV row per (backend, scene).
#[expect(
    clippy::too_many_arguments,
    reason = "internal driver, not a public API"
)]
fn run_robot<R, const N: usize>(
    robot: R,
    robot_name: &str,
    joint_names: &[&str; N],
    tf: Isometry3<f32>,
    r_range: (f32, f32),
    mvt_cpp_r_range: (f32, f32),
    voxel_width: f32,
    resources: &Path,
    datasets: &[&str],
    csv: &mut impl Write,
) -> Result<(), Box<dyn std::error::Error>>
where
    R: Robot<N, f32>
        + BlockValidate<N, f32, World3d<f32>>
        + BlockValidate<N, f32, PointCloudWorld<Mvt<3, f32>>>
        + BlockValidate<N, f32, PointCloudWorld<MutableMvt<3, f32>>>
        + BlockValidate<N, f32, PointCloudWorld<MvtCpp>>
        + BlockValidate<N, f32, PointCloudWorld<Capt<3, f32, u32>>>
        + BlockValidate<N, f32, PointCloudWorld<ImmutableKdTree<f32, 3>>>
        + BlockValidate<N, f32, SimdPointCloudWorld<Mvt<3, f32>>>
        + BlockValidate<N, f32, SimdPointCloudWorld<MutableMvt<3, f32>>>
        + BlockValidate<N, f32, SimdPointCloudWorld<MvtCpp>>
        + BlockValidate<N, f32, SimdPointCloudWorld<Capt<3, f32, u32>>>
        + Clone,
{
    let r_filter = R_FILTER_SCALE * r_range.0;
    let cap = max_problems_per_dataset();

    for &dataset in datasets {
        let prob_dir = resources
            .join(robot_name)
            .join("problems")
            .join(format!("{dataset}_{robot_name}"));
        let problems = dir_to_problems(&prob_dir, joint_names, tf)?;

        // The first `cap` usable scenes, with their sampled and filtered point clouds.
        let scenes: Vec<(&Problem<N>, usize, Vec<[f32; 3]>)> = problems
            .iter()
            .filter(|p| is_sampleable(&p.world))
            .filter_map(|problem| {
                let mut rng = SmallRng::seed_from_u64(problem.id as u64);
                let full_points = sample_scene(&problem.world, DENSITY, &mut rng);
                let filtered_points = centervox_filter(&full_points, r_filter);
                (!filtered_points.is_empty()).then_some((
                    problem,
                    full_points.len(),
                    filtered_points,
                ))
            })
            .take(cap)
            .collect();
        let build_times = BuildTimes::measure(&scenes, voxel_width, r_range, mvt_cpp_r_range);

        for (scene_idx, (problem, n_full_points, filtered_points)) in scenes.iter().enumerate() {
            let problem = *problem;
            println!(
                "{robot_name}/{dataset}#{}: {} points -> {} filtered (r_filter={r_filter:.4})",
                problem.id,
                n_full_points,
                filtered_points.len(),
            );

            run_one_backend(
                &robot,
                robot_name,
                dataset,
                problem,
                filtered_points,
                r_range,
                r_filter,
                "primitive",
                build_times.primitive[scene_idx],
                |_pc, _| Ok(problem.world.clone()),
                csv,
            )?;
            run_one_backend(
                &robot,
                robot_name,
                dataset,
                problem,
                filtered_points,
                r_range,
                r_filter,
                "mvtable",
                build_times.mvtable[scene_idx],
                |pc, _| Ok(PointCloudWorld(Mvt::new(pc, voxel_width))),
                csv,
            )?;
            run_one_backend(
                &robot,
                robot_name,
                dataset,
                problem,
                filtered_points,
                r_range,
                r_filter,
                "mvtable_mutable",
                build_times.mvtable_mutable[scene_idx],
                |pc, _| Ok(PointCloudWorld(MutableMvt::new(pc, voxel_width))),
                csv,
            )?;
            run_one_backend(
                &robot,
                robot_name,
                dataset,
                problem,
                filtered_points,
                mvt_cpp_r_range,
                r_filter,
                "mvtable_cpp",
                build_times.mvt_cpp[scene_idx],
                |pc, r_range| {
                    MvtCpp::try_new(pc, r_range)
                        .map(PointCloudWorld)
                        .map_err(Into::into)
                },
                csv,
            )?;
            run_one_backend(
                &robot,
                robot_name,
                dataset,
                problem,
                filtered_points,
                r_range,
                r_filter,
                "capt",
                build_times.capt[scene_idx],
                |pc, r_range| Ok(PointCloudWorld(Capt::new(pc, r_range, 1))),
                csv,
            )?;
            run_one_backend(
                &robot,
                robot_name,
                dataset,
                problem,
                filtered_points,
                r_range,
                r_filter,
                "kiddo",
                build_times.kiddo[scene_idx],
                |pc, _| {
                    ImmutableKdTree::new_from_slice(pc)
                        .map(PointCloudWorld)
                        .map_err(Into::into)
                },
                csv,
            )?;
            run_one_backend(
                &robot,
                robot_name,
                dataset,
                problem,
                filtered_points,
                r_range,
                r_filter,
                "mvtable_simd",
                build_times.mvtable[scene_idx],
                |pc, _| Ok(SimdPointCloudWorld(Mvt::new(pc, voxel_width))),
                csv,
            )?;
            run_one_backend(
                &robot,
                robot_name,
                dataset,
                problem,
                filtered_points,
                r_range,
                r_filter,
                "mvtable_mutable_simd",
                build_times.mvtable_mutable[scene_idx],
                |pc, _| Ok(SimdPointCloudWorld(MutableMvt::new(pc, voxel_width))),
                csv,
            )?;
            run_one_backend(
                &robot,
                robot_name,
                dataset,
                problem,
                filtered_points,
                mvt_cpp_r_range,
                r_filter,
                "mvt_cpp_simd",
                build_times.mvt_cpp[scene_idx],
                |pc, r_range| {
                    MvtCpp::try_new(pc, r_range)
                        .map(SimdPointCloudWorld)
                        .map_err(Into::into)
                },
                csv,
            )?;
            run_one_backend(
                &robot,
                robot_name,
                dataset,
                problem,
                filtered_points,
                r_range,
                r_filter,
                "capt_simd",
                build_times.capt_simd[scene_idx],
                |pc, _| Ok(SimdPointCloudWorld(Capt::new(pc, r_range, 8))),
                csv,
            )?;
        }

        if scenes.is_empty() {
            eprintln!("skipping {robot_name}/{dataset}: no usable scenes found");
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let resources = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../resources");
    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data");
    fs::create_dir_all(&data_dir)?;

    let mut csv = BufWriter::new(File::create(data_dir.join("mbm_plan_results.csv"))?);
    writeln!(
        csv,
        "structure,robot,dataset,scene_id,r_filter,n_points,solved,time_secs,construction_secs,\
         n_samples,n_nodes"
    )?;

    run_robot(
        Panda,
        "panda",
        &Panda::JOINT_NAMES,
        Isometry3::identity(),
        (Panda::MIN_RADIUS, mvtable_bench::mobile_max_radius("panda")),
        (
            Panda::MIN_RADIUS,
            mvtable_bench::true_max_query_radius("panda"),
        ),
        PANDA_VOXEL_WIDTH,
        &resources,
        &DATASETS,
        &mut csv,
    )?;
    run_robot(
        Ur5,
        "ur5",
        &Ur5::JOINT_NAMES,
        Isometry3::new(
            Vector3::new(0.0, 0.0, -0.9144),
            Vector3::new(0.0, 0.0, -1.57),
        ),
        (Ur5::MIN_RADIUS, mvtable_bench::mobile_max_radius("ur5")),
        (Ur5::MIN_RADIUS, mvtable_bench::true_max_query_radius("ur5")),
        UR5_VOXEL_WIDTH,
        &resources,
        &DATASETS,
        &mut csv,
    )?;
    run_robot(
        Fetch,
        "fetch",
        &Fetch::JOINT_NAMES,
        Isometry3::identity(),
        (Fetch::MIN_RADIUS, mvtable_bench::mobile_max_radius("fetch")),
        (
            Fetch::MIN_RADIUS,
            mvtable_bench::true_max_query_radius("fetch"),
        ),
        FETCH_VOXEL_WIDTH,
        &resources,
        &DATASETS,
        &mut csv,
    )?;
    run_robot(
        Baxter,
        "baxter",
        &Baxter::JOINT_NAMES,
        Isometry3::identity(),
        (
            Baxter::MIN_RADIUS,
            mvtable_bench::mobile_max_radius("baxter"),
        ),
        (
            Baxter::MIN_RADIUS,
            mvtable_bench::true_max_query_radius("baxter"),
        ),
        BAXTER_VOXEL_WIDTH,
        &resources,
        &BAXTER_DATASETS,
        &mut csv,
    )?;

    Ok(())
}
