//! Extracts realistic point-cloud collision-checking workloads (point clouds + query traces) from
//! the MotionBenchMaker dataset, by running a planner over a sample of scenes from each
//! MotionBenchMaker benchmark environment, for every robot MotionBenchMaker has problem sets for
//! (Panda, UR5, Fetch, Baxter).
//!
//! For performance, each planner run does its collision checking directly against the scene's
//! ground-truth geometry. Every collision query issued during planning is recorded, alongside the
//! ground-truth collision result it got.
//!
//! Output goes to `data/raw/<robot>_<dataset>_<scene_id>_{points_full,queries}.bin` plus a
//! top-level `data/manifest.csv`, all at the workspace root. Point cloud and query files use a
//! minimal binary format (a `u64` little-endian record count, followed by fixed-width
//! little-endian records).
//!
//! The MotionBenchMaker scene/request YAML files this reads from are vendored under
//! `resources/` (copied from the `rumple` git checkout's own `resources/<robot>/problems/`
//! directories - only the problem sets, not the meshes/URDFs, which extraction never touches),
//! rather than located at runtime, so this binary doesn't depend on a particular local cargo
//! git-checkout layout to run.

use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use carom::{
    BlockValidate, Robot,
    robot::{Baxter, Fetch, Panda, Ur5},
};
use mbm::dir_to_problems;
use mbm_extract::{RecordedQuery, RecordingWorld, sample_scene, solve_recording};
use nalgebra::{Isometry3, Vector3};
use rand::{SeedableRng, rngs::SmallRng};

/// The 7 canonical MotionBenchMaker benchmark environments, Used by
/// every robot except Baxter, which ships a different problem set (see [`BAXTER_DATASETS`]).
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

/// Maximum number of scene indices to extract per MotionBenchMaker problem directory. Each
/// directory holds on the order of 100 scenes (600 for Baxter), and that is a little too much.
const MAX_PROBLEMS_PER_DATASET: usize = 20;

/// Surface-sample density (points per unit area) used to convert scene geometry into a point
/// cloud; tuned so full point clouds land in roughly the 5,000-50,000 point range.
const DENSITY: f32 = 6000.0;

/// Write `points` as a `u64` little-endian count followed by that many `[f32; 3]` records (12
/// bytes each, little-endian).
fn write_points(path: impl AsRef<Path>, points: &[[f32; 3]]) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(&(points.len() as u64).to_le_bytes())?;
    for [x, y, z] in points {
        w.write_all(&x.to_le_bytes())?;
        w.write_all(&y.to_le_bytes())?;
        w.write_all(&z.to_le_bytes())?;
    }
    Ok(())
}

/// Write `queries` as a `u64` little-endian count followed by that many
/// `(x, y, z, r, collided, lanes)` records (4 little-endian `f32`s + 2 `u8`s, 18 bytes each).
/// `lanes` is the width of the original `collides_balls::<L>` call each query was part of (see
/// [`RecordedQuery::lanes`]) and lets replay reconstruct the planner's true SIMD batch structure.
fn write_queries(path: impl AsRef<Path>, queries: &[RecordedQuery]) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(&(queries.len() as u64).to_le_bytes())?;
    for q in queries {
        w.write_all(&q.x.to_le_bytes())?;
        w.write_all(&q.y.to_le_bytes())?;
        w.write_all(&q.z.to_le_bytes())?;
        w.write_all(&q.r.to_le_bytes())?;
        w.write_all(&[u8::from(q.collided), q.lanes])?;
    }
    Ok(())
}

/// Whether [`sample_scene`] can convert every primitive in `world` into a point cloud (it only
/// handles boxes, rotated cuboids, and cylinders).
fn is_sampleable(world: &carom::env::World3d<f32>) -> bool {
    world.aabbs.is_empty() && world.balls.is_empty() && world.point_clouds.is_empty()
}

/// Extract a point cloud + query trace from up to [`MAX_PROBLEMS_PER_DATASET`] scenes per
/// dataset, for a single robot, appending a manifest row for each scene actually extracted.
#[expect(
    clippy::too_many_arguments,
    reason = "internal driver, not a public API"
)]
fn extract_robot<R, const N: usize>(
    robot: R,
    robot_name: &str,
    joint_names: &[&str; N],
    tf: Isometry3<f32>,
    resources: &Path,
    raw_dir: &Path,
    datasets: &[&str],
    manifest: &mut impl Write,
) -> Result<(), Box<dyn std::error::Error>>
where
    R: Robot<N, f32> + BlockValidate<N, f32, RecordingWorld> + Clone,
{
    for &dataset in datasets {
        let prob_dir = resources
            .join(robot_name)
            .join("problems")
            .join(format!("{dataset}_{robot_name}"));
        let problems = dir_to_problems(&prob_dir, joint_names, tf)?;

        let mut n_extracted = 0usize;
        for problem in problems.iter().filter(|p| is_sampleable(&p.world)) {
            if n_extracted >= MAX_PROBLEMS_PER_DATASET {
                break;
            }

            let result = match solve_recording(
                robot.clone(),
                problem,
                RecordingWorld::new(problem.world.clone()),
            ) {
                Ok(result) => result,
                Err(e) => {
                    eprintln!("skipping {robot_name}/{dataset}#{}: {e}", problem.id);
                    continue;
                }
            };

            let mut rng = SmallRng::seed_from_u64(problem.id as u64);
            let full_points = sample_scene(&problem.world, DENSITY, &mut rng);
            if full_points.is_empty() {
                continue;
            }
            let n_points_full = full_points.len();

            let prefix = format!("{robot_name}_{dataset}_{}", problem.id);
            write_points(
                raw_dir.join(format!("{prefix}_points_full.bin")),
                &full_points,
            )?;
            write_queries(
                raw_dir.join(format!("{prefix}_queries.bin")),
                &result.queries,
            )?;

            let n_collided = result.queries.iter().filter(|q| q.collided).count();
            let solved = result.status.trajectory.is_some();
            println!(
                "{robot_name}/{dataset}#{}: {n_points_full} points, {} queries ({n_collided} \
                 collided), solved={solved}, {:.3}s",
                problem.id,
                result.queries.len(),
                result.status.time.as_secs_f64(),
            );

            writeln!(
                manifest,
                "{dataset},{robot_name},{},{n_points_full},{},{n_collided},{solved}",
                problem.id,
                result.queries.len(),
            )?;
            manifest.flush()?;

            n_extracted += 1;
        }

        if n_extracted == 0 {
            eprintln!("skipping {robot_name}/{dataset}: no usable scenes found");
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let resources = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../resources");
    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let raw_dir = data_dir.join("raw");
    fs::create_dir_all(&raw_dir)?;

    let mut manifest = BufWriter::new(File::create(data_dir.join("manifest.csv"))?);
    writeln!(
        manifest,
        "dataset,robot,scene_id,n_points_full,n_queries,n_collided,solved"
    )?;

    extract_robot(
        Panda,
        "panda",
        &Panda::JOINT_NAMES,
        Isometry3::identity(),
        &resources,
        &raw_dir,
        &DATASETS,
        &mut manifest,
    )?;
    extract_robot(
        Ur5,
        "ur5",
        &Ur5::JOINT_NAMES,
        Isometry3::new(
            Vector3::new(0.0, 0.0, -0.9144),
            Vector3::new(0.0, 0.0, -1.57),
        ),
        &resources,
        &raw_dir,
        &DATASETS,
        &mut manifest,
    )?;
    extract_robot(
        Fetch,
        "fetch",
        &Fetch::JOINT_NAMES,
        Isometry3::identity(),
        &resources,
        &raw_dir,
        &DATASETS,
        &mut manifest,
    )?;
    extract_robot(
        Baxter,
        "baxter",
        &Baxter::JOINT_NAMES,
        Isometry3::identity(),
        &resources,
        &raw_dir,
        &BAXTER_DATASETS,
        &mut manifest,
    )?;

    Ok(())
}
