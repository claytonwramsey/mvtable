//! End-to-end motion-planning benchmark.
//!
//! Output goes to `data/mbm_plan_results.csv` at the workspace root.

use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use capt::Capt;
use carom::{
    BlockValidate, Robot,
    robot::{Baxter, Fetch, Panda, Ur5},
};
use kiddo::ImmutableKdTree;
use mbm::{Problem, dir_to_problems};
use mbm_plan_bench::{
    PointCloudWorld, SimdPointCloudWorld, is_sampleable, sample_scene, solve_with_backend,
};
use mvtable::{MutableMvt, Mvt};
use mvtable_bench::filter::centervox_filter;
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
const PANDA_VOXEL_WIDTH: f32 = 0.17324;
const UR5_VOXEL_WIDTH: f32 = 0.22427;
const FETCH_VOXEL_WIDTH: f32 = 0.13799;
const BAXTER_VOXEL_WIDTH: f32 = 0.17075;

/// Wall-clock cutoff per (backend, problem) solve attempt, so a pathologically slow combination
/// can't stall the whole unattended run.
const MAX_SOLVE_TIME: Duration = Duration::from_secs(10);

fn max_problems_per_dataset() -> usize {
    std::env::var("MBM_PLAN_BENCH_MAX_SCENES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(MAX_PROBLEMS_PER_DATASET)
}

/// Build `structure_name`'s backend from `filtered_points`, run the planner against it, and write
/// one CSV row with the outcome (or log and return `Ok(())` without writing a row, if this
/// particular backend/problem combination fails to solve).
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
    builder: impl Fn(&[[f32; 3]], (f32, f32)) -> W,
    csv: &mut impl Write,
) -> Result<(), Box<dyn std::error::Error>>
where
    R: Robot<N, f32> + BlockValidate<N, f32, W> + Clone,
{
    let tic = Instant::now();
    let structure = builder(filtered_points, r_range);
    let construction_secs = tic.elapsed().as_secs_f64();

    let result = match solve_with_backend(robot.clone(), problem, structure, MAX_SOLVE_TIME) {
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
    voxel_width: f32,
    resources: &Path,
    datasets: &[&str],
    csv: &mut impl Write,
) -> Result<(), Box<dyn std::error::Error>>
where
    R: Robot<N, f32>
        + BlockValidate<N, f32, PointCloudWorld<Mvt<3, f32>>>
        + BlockValidate<N, f32, PointCloudWorld<MutableMvt<3, f32>>>
        + BlockValidate<N, f32, PointCloudWorld<Capt<3, f32, u32>>>
        + BlockValidate<N, f32, PointCloudWorld<ImmutableKdTree<f32, 3>>>
        + BlockValidate<N, f32, SimdPointCloudWorld<Mvt<3, f32>>>
        + BlockValidate<N, f32, SimdPointCloudWorld<MutableMvt<3, f32>>>
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

        let mut n_run = 0usize;
        for problem in problems.iter().filter(|p| is_sampleable(&p.world)) {
            if n_run >= cap {
                break;
            }

            let mut rng = SmallRng::seed_from_u64(problem.id as u64);
            let full_points = sample_scene(&problem.world, DENSITY, &mut rng);
            if full_points.is_empty() {
                continue;
            }
            let filtered_points = centervox_filter(&full_points, r_filter);
            if filtered_points.is_empty() {
                continue;
            }

            println!(
                "{robot_name}/{dataset}#{}: {} points -> {} filtered (r_filter={r_filter:.4})",
                problem.id,
                full_points.len(),
                filtered_points.len(),
            );

            run_one_backend(
                &robot,
                robot_name,
                dataset,
                problem,
                &filtered_points,
                r_range,
                r_filter,
                "mvtable",
                |pc, _| PointCloudWorld(Mvt::new(pc, voxel_width)),
                csv,
            )?;
            run_one_backend(
                &robot,
                robot_name,
                dataset,
                problem,
                &filtered_points,
                r_range,
                r_filter,
                "mvtable_mutable",
                |pc, _| PointCloudWorld(MutableMvt::new(pc, voxel_width)),
                csv,
            )?;
            run_one_backend(
                &robot,
                robot_name,
                dataset,
                problem,
                &filtered_points,
                r_range,
                r_filter,
                "capt",
                |pc, r_range| PointCloudWorld(Capt::new(pc, r_range, 1)),
                csv,
            )?;
            run_one_backend(
                &robot,
                robot_name,
                dataset,
                problem,
                &filtered_points,
                r_range,
                r_filter,
                "kiddo",
                |pc, _| PointCloudWorld(ImmutableKdTree::new_from_slice(pc)),
                csv,
            )?;
            run_one_backend(
                &robot,
                robot_name,
                dataset,
                problem,
                &filtered_points,
                r_range,
                r_filter,
                "mvtable_simd",
                |pc, _| SimdPointCloudWorld(Mvt::new(pc, voxel_width)),
                csv,
            )?;
            run_one_backend(
                &robot,
                robot_name,
                dataset,
                problem,
                &filtered_points,
                r_range,
                r_filter,
                "mvtable_mutable_simd",
                |pc, _| SimdPointCloudWorld(MutableMvt::new(pc, voxel_width)),
                csv,
            )?;
            run_one_backend(
                &robot,
                robot_name,
                dataset,
                problem,
                &filtered_points,
                r_range,
                r_filter,
                "capt_simd",
                |pc, _| SimdPointCloudWorld(Capt::new(pc, r_range, 8)),
                csv,
            )?;

            n_run += 1;
        }

        if n_run == 0 {
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
        (Baxter::MIN_RADIUS, mvtable_bench::mobile_max_radius("baxter")),
        BAXTER_VOXEL_WIDTH,
        &resources,
        &BAXTER_DATASETS,
        &mut csv,
    )?;

    Ok(())
}
