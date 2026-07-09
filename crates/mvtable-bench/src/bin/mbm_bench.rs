//! Replays point-cloud collision-checking workloads extracted from the MotionBenchMaker
//! dataset (see `crates/mbm-extract`) against `mvtable`, `capt`, and `kiddo`.
//!
//! Writes `data/mbm_bench_results.csv`, which `scripts/plot_mbm.py` turns into a throughput
//! figure.
#![feature(portable_simd)]

use std::{
    collections::HashMap,
    error::Error,
    fs::File,
    hint::black_box,
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    simd::Simd,
    time::Instant,
};

use capt::AxisSimd;
use mvtable_bench::{SimdStructure, Structure, filter};
use rand::{SeedableRng, rngs::SmallRng};

/// Maximum number of queries to try against a single trace (to prevent trace replay from taking too
/// long).
const MAX_QUERIES: usize = 10_000;

/// SIMD lane width benchmarked for `mvtable` and `capt`. Also the lane count `capt` is
/// constructed with, so that the same instance serves both the scalar and SIMD benchmarks.
const SIMD_L: usize = 8;

/// Filters compared, by name; see [`apply_filter`].
const FILTER_NAMES: [&str; 2] = ["centervox", "morton"];

/// Schedule of filter resolutions (voxel size / minimum separation), each a multiple
/// of a workload's own smallest queried sphere radius `r_min`.
const FILTER_RADIUS_SCALES: [f32; 5] = [4.0, 8.0, 16.0, 32.0, 64.0];

struct Query {
    center: [f32; 3],
    r: f32,
}

/// Read a point cloud written by `mbm-extract`'s `write_points`: a `u64` little-endian count
/// followed by that many `[f32; 3]` little-endian records.
fn read_points(path: impl AsRef<Path>) -> Result<Vec<[f32; 3]>, Box<dyn Error>> {
    let mut r = BufReader::new(File::open(path)?);
    let mut count_buf = [0u8; 8];
    r.read_exact(&mut count_buf)?;
    let count = u64::from_le_bytes(count_buf) as usize;

    let mut points = Vec::with_capacity(count);
    let mut buf = [0u8; 12];
    for _ in 0..count {
        r.read_exact(&mut buf)?;
        points.push([
            f32::from_le_bytes(buf[0..4].try_into().unwrap()),
            f32::from_le_bytes(buf[4..8].try_into().unwrap()),
            f32::from_le_bytes(buf[8..12].try_into().unwrap()),
        ]);
    }
    Ok(points)
}

/// Read a query trace written by `mbm-extract`'s `write_queries`: a `u64` little-endian count
/// followed by that many `(x, y, z, r, collided)` little-endian records.
fn read_queries(path: impl AsRef<Path>) -> Result<Vec<Query>, Box<dyn Error>> {
    let mut r = BufReader::new(File::open(path)?);
    let mut count_buf = [0u8; 8];
    r.read_exact(&mut count_buf)?;
    let count = u64::from_le_bytes(count_buf) as usize;

    let mut queries = Vec::with_capacity(count);
    let mut buf = [0u8; 17];
    for _ in 0..count {
        r.read_exact(&mut buf)?;
        queries.push(Query {
            center: [
                f32::from_le_bytes(buf[0..4].try_into().unwrap()),
                f32::from_le_bytes(buf[4..8].try_into().unwrap()),
                f32::from_le_bytes(buf[8..12].try_into().unwrap()),
            ],
            r: f32::from_le_bytes(buf[12..16].try_into().unwrap()),
        });
    }
    Ok(queries)
}

/// Downsample `points` with the named filter (one of [`FILTER_NAMES`]) at the given `resolution`
/// (voxel size / minimum separation).
fn apply_filter(name: &str, points: &[[f32; 3]], resolution: f32) -> Vec<[f32; 3]> {
    match name {
        "centervox" => filter::centervox_filter(points, resolution),
        "morton" => {
            let mut points = points.to_vec();
            filter::morton_filter(&mut points, resolution);
            points
        }
        _ => unreachable!("FILTER_NAMES only contains centervox and morton"),
    }
}

/// Deterministically subsample `queries` down to `size` (or return them unchanged if already at
/// most `size`), seeded by `seed` so runs are reproducible.
fn subsample_queries(queries: Vec<Query>, size: usize, seed: u64) -> Vec<Query> {
    if queries.len() <= size {
        return queries;
    }
    let mut rng = SmallRng::seed_from_u64(seed);
    let idx = rand::seq::index::sample(&mut rng, queries.len(), size);
    idx.into_iter()
        .map(|i| Query {
            center: queries[i].center,
            r: queries[i].r,
        })
        .collect()
}

/// Group `queries` into batches of `L`, converting each batch into a SIMD center/radius vector.
/// Any remainder that doesn't fill a full batch of `L` is dropped.
fn to_simd_batches<const L: usize>(queries: &[&Query]) -> Vec<([Simd<f32, L>; 3], Simd<f32, L>)> {
    queries
        .chunks_exact(L)
        .map(|chunk| {
            let centers: [Simd<f32, L>; 3] = std::array::from_fn(|k| {
                Simd::from_array(std::array::from_fn(|lane| chunk[lane].center[k]))
            });
            let radii = Simd::from_array(std::array::from_fn(|lane| chunk[lane].r));
            (centers, radii)
        })
        .collect()
}

/// Time replaying every query in `queries` against `structure`.
fn time_queries<S: Structure<3>>(structure: &S, queries: &[&Query]) -> std::time::Duration {
    let tic = Instant::now();
    for q in queries {
        black_box(structure.collides(&q.center, q.r));
    }
    tic.elapsed()
}

/// Time replaying every SIMD batch in `batches` against `structure`.
fn time_simd_queries<S: SimdStructure<3>, const L: usize>(
    structure: &S,
    batches: &[([Simd<f32, L>; 3], Simd<f32, L>)],
) -> std::time::Duration
where
    Simd<f32, L>: AxisSimd<L>,
{
    let tic = Instant::now();
    for (centers, radii) in batches {
        black_box(structure.collides_simd(centers, *radii));
    }
    tic.elapsed()
}

/// Split `queries` into `(all, colliding, non_colliding)` reference traces (using the precomputed
/// `colliding`/`non_colliding` index sets), skip empty traces, and call `record` with each
/// non-empty `(trace_name, trace)` pair.
fn for_each_trace<'q>(
    queries: &'q [Query],
    colliding: &[usize],
    non_colliding: &[usize],
    mut record: impl FnMut(&str, &[&'q Query]) -> Result<(), Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    let all: Vec<&Query> = queries.iter().collect();
    let colliding: Vec<&Query> = colliding.iter().map(|&i| &queries[i]).collect();
    let non_colliding: Vec<&Query> = non_colliding.iter().map(|&i| &queries[i]).collect();
    for (trace_name, trace) in [
        ("all", all),
        ("colliding", colliding),
        ("non_colliding", non_colliding),
    ] {
        if !trace.is_empty() {
            record(trace_name, &trace)?;
        }
    }
    Ok(())
}

/// Fields shared by every row `bench_scalar`/`bench_simd`/`write_construction_row` write, bundled
/// up so those functions don't each need half a dozen near-identical parameters.
#[derive(Clone, Copy)]
struct RowContext<'a> {
    structure: &'a str,
    workload: &'a str,
    filter_name: &'a str,
    n_points: usize,
}

fn bench_scalar<S: Structure<3>>(
    out: &mut impl Write,
    ctx: RowContext,
    structure: &S,
    queries: &[Query],
    colliding: &[usize],
    non_colliding: &[usize],
) -> Result<(), Box<dyn Error>> {
    for_each_trace(queries, colliding, non_colliding, |trace_name, trace| {
        let elapsed_ns = time_queries(structure, trace).as_secs_f64() * 1e9;
        let ns_per_query = elapsed_ns / trace.len() as f64;
        writeln!(
            out,
            "{},{},{},{},{},{trace_name},1,{ns_per_query}",
            ctx.structure,
            ctx.workload,
            ctx.filter_name,
            ctx.n_points,
            trace.len()
        )?;
        Ok(())
    })
}

fn bench_simd<S: SimdStructure<3>>(
    out: &mut impl Write,
    ctx: RowContext,
    structure: &S,
    queries: &[Query],
    colliding: &[usize],
    non_colliding: &[usize],
) -> Result<(), Box<dyn Error>> {
    for_each_trace(queries, colliding, non_colliding, |trace_name, trace| {
        if trace.len() < SIMD_L {
            return Ok(());
        }
        let batches = to_simd_batches::<SIMD_L>(trace);
        let n_simd_queries = batches.len() * SIMD_L;
        let elapsed_ns = time_simd_queries(structure, &batches).as_secs_f64() * 1e9;
        let ns_per_query = elapsed_ns / n_simd_queries as f64;
        writeln!(
            out,
            "{},{},{},{},{n_simd_queries},{trace_name},{SIMD_L},{ns_per_query}",
            ctx.structure, ctx.workload, ctx.filter_name, ctx.n_points,
        )?;
        Ok(())
    })
}

fn write_construction_row(
    out: &mut impl Write,
    ctx: RowContext,
    n_queries: usize,
    ns: f64,
) -> Result<(), Box<dyn Error>> {
    writeln!(
        out,
        "{},{},{},{},{n_queries},construction,1,{ns}",
        ctx.structure, ctx.workload, ctx.filter_name, ctx.n_points,
    )?;
    Ok(())
}

/// Record `bytes` (from a structure's `memory_used()`) as a `memory` metric row.
fn write_memory_row(
    out: &mut impl Write,
    ctx: RowContext,
    bytes: usize,
) -> Result<(), Box<dyn Error>> {
    writeln!(
        out,
        "{},{},{},{},0,memory,1,{bytes}",
        ctx.structure, ctx.workload, ctx.filter_name, ctx.n_points,
    )?;
    Ok(())
}

/// A `(dataset, robot, scene_id)` triple from `data/manifest.csv`, identifying one extracted
/// point cloud + query trace.
struct Workload {
    dataset: String,
    robot: String,
    scene_id: String,
}

impl Workload {
    /// The filename prefix used for this workload's `_points_full.bin` / `_queries.bin` files
    /// under `data/raw/`.
    fn file_prefix(&self) -> String {
        format!("{}_{}_{}", self.robot, self.dataset, self.scene_id)
    }

    /// A human-readable, robot- and scene-qualified label used in the results CSV.
    fn label(&self) -> String {
        format!("{}/{}/{}", self.robot, self.dataset, self.scene_id)
    }
}

/// Read `data/manifest.csv`, optionally capping how many scenes are kept per (robot, dataset)
/// group at `max_per_dataset` (see `MBM_BENCH_MAX_SCENES`).
fn read_workloads(
    data_dir: &Path,
    max_per_dataset: Option<usize>,
) -> Result<Vec<Workload>, Box<dyn Error>> {
    let manifest = BufReader::new(File::open(data_dir.join("manifest.csv"))?);
    let mut seen: HashMap<(String, String), usize> = HashMap::new();
    let mut workloads = Vec::new();

    for line in manifest.lines().skip(1) {
        let line = line?;
        let mut cols = line.split(',');
        let dataset = cols
            .next()
            .ok_or("manifest row missing dataset column")?
            .to_owned();
        let robot = cols
            .next()
            .ok_or("manifest row missing robot column")?
            .to_owned();
        let scene_id = cols
            .next()
            .ok_or("manifest row missing scene_id column")?
            .to_owned();

        if let Some(max) = max_per_dataset {
            let count = seen.entry((robot.clone(), dataset.clone())).or_insert(0);
            if *count >= max {
                continue;
            }
            *count += 1;
        }

        workloads.push(Workload {
            dataset,
            robot,
            scene_id,
        });
    }

    Ok(workloads)
}

fn main() -> Result<(), Box<dyn Error>> {
    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data");
    let raw_dir = data_dir.join("raw");
    let max_scenes = std::env::var("MBM_BENCH_MAX_SCENES")
        .ok()
        .and_then(|s| s.parse().ok());
    let workloads = read_workloads(&data_dir, max_scenes)?;

    let out_path = data_dir.join("mbm_bench_results.csv");
    let mut out = BufWriter::new(File::create(&out_path)?);
    writeln!(
        out,
        "structure,dataset,filter,n_points,n_queries,metric,lanes,ns_per_op"
    )?;

    for workload in &workloads {
        let prefix = workload.file_prefix();
        let label = workload.label();
        let full_points = read_points(raw_dir.join(format!("{prefix}_points_full.bin")))?;
        let all_queries = read_queries(raw_dir.join(format!("{prefix}_queries.bin")))?;
        let r_max = all_queries.iter().fold(0.0f32, |m, q| m.max(q.r));
        let r_min = all_queries.iter().fold(f32::INFINITY, |m, q| m.min(q.r));
        let r_range = (r_min, r_max);

        // Subsampled once per workload (not per filter/radius), so every (filter, radius)
        // combination below is compared on the exact same query set.
        let seed = prefix
            .bytes()
            .fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(b as u64));
        let queries = subsample_queries(all_queries, MAX_QUERIES, seed);

        for &filter_name in &FILTER_NAMES {
            for &scale in &FILTER_RADIUS_SCALES {
                // Points closer together than the robot's smallest queried sphere add no useful
                // collision-checking resolution, so that's a natural, workload-specific base
                // filter strength; the exponential schedule sweeps around it.
                let points = apply_filter(filter_name, &full_points, scale * r_min);
                let n_points = points.len();
                if n_points == 0 {
                    continue;
                }

                // Ground-truth colliding/non-colliding partition for this point cloud, computed
                // with `mvtable` itself.
                let oracle = mvtable::Mvt::<3, f32>::new(&points, r_max);
                let mut colliding = Vec::new();
                let mut non_colliding = Vec::new();
                for (i, q) in queries.iter().enumerate() {
                    if Structure::collides(&oracle, &q.center, q.r) {
                        colliding.push(i);
                    } else {
                        non_colliding.push(i);
                    }
                }

                println!(
                    "{label} [{filter_name} x{scale}] @ {n_points} points (from \
                     {}): {} queries ({} colliding, {} non-colliding)",
                    full_points.len(),
                    queries.len(),
                    colliding.len(),
                    non_colliding.len()
                );

                // `mvtable`: one instance, reused for scalar and SIMD queries (construction
                // doesn't depend on the SIMD lane width, unlike `capt`).
                let ctx = RowContext {
                    structure: "mvtable",
                    workload: &label,
                    filter_name,
                    n_points,
                };
                let tic = Instant::now();
                let mvt = mvtable::Mvt::<3, f32>::new(&points, r_max);
                write_construction_row(
                    &mut out,
                    ctx,
                    queries.len(),
                    tic.elapsed().as_secs_f64() * 1e9,
                )?;
                write_memory_row(&mut out, ctx, mvt.memory_used())?;
                bench_scalar(&mut out, ctx, &mvt, &queries, &colliding, &non_colliding)?;
                bench_simd(&mut out, ctx, &mvt, &queries, &colliding, &non_colliding)?;

                // `capt`: built once at `SIMD_L` lanes (rather than once at 1 lane for scalar and
                // again at `SIMD_L` lanes for SIMD), reused for both benchmarks below.
                let ctx = RowContext {
                    structure: "capt",
                    ..ctx
                };
                let tic = Instant::now();
                let capt = capt::Capt::<3, f32, u32>::new(&points, r_range, SIMD_L);
                write_construction_row(
                    &mut out,
                    ctx,
                    queries.len(),
                    tic.elapsed().as_secs_f64() * 1e9,
                )?;
                write_memory_row(&mut out, ctx, capt.memory_used())?;
                bench_scalar(&mut out, ctx, &capt, &queries, &colliding, &non_colliding)?;
                bench_simd(&mut out, ctx, &capt, &queries, &colliding, &non_colliding)?;

                // `kiddo`: scalar only, no SIMD-batched query API.
                let ctx = RowContext {
                    structure: "kiddo",
                    ..ctx
                };
                let tic = Instant::now();
                let kdt = kiddo::ImmutableKdTree::<f32, 3>::new_from_slice(&points);
                write_construction_row(
                    &mut out,
                    ctx,
                    queries.len(),
                    tic.elapsed().as_secs_f64() * 1e9,
                )?;
                bench_scalar(&mut out, ctx, &kdt, &queries, &colliding, &non_colliding)?;

                out.flush()?;
            }
        }
    }

    println!("wrote {}", out_path.display());

    Ok(())
}
