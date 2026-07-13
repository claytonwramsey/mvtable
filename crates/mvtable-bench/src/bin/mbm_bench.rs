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
    simd::{Simd, cmp::SimdPartialEq},
    time::Instant,
};

use capt::AxisSimd;
use mvtable_bench::{SimdStructure, Structure, filter};
use rand::{SeedableRng, rngs::SmallRng};

/// Maximum number of queries to try against the scalar/sequential trace for a single workload
/// (to prevent trace replay from taking too long).
const MAX_QUERIES: usize = 10_000;

/// Maximum number of SIMD batches to try against the parallel trace for a single workload.
/// Chosen so the total number of underlying scalar-equivalent queries replayed
/// (`MAX_SIMD_BATCHES * SIMD_L`) stays the same order of magnitude as [`MAX_QUERIES`].
const MAX_SIMD_BATCHES: usize = MAX_QUERIES / SIMD_L;

/// SIMD lane width benchmarked for `mvtable` and `capt`. Also the lane count `capt` is
/// constructed with, so that the same instance serves both the scalar and SIMD benchmarks. This
/// must match the lane width `mbm-extract`'s `carom::Rake<_, _, _, 8>` actually replays raked
/// motion-segment validity checks with (see `RawQuery::lanes`).
const SIMD_L: usize = 8;

/// Filters compared, by name; see [`apply_filter`].
const FILTER_NAMES: [&str; 2] = ["centervox", "morton"];

/// Schedule of filter resolutions (voxel size / minimum separation), each a multiple
/// of a workload's own smallest queried sphere radius `r_min`.
const FILTER_RADIUS_SCALES: [f32; 5] = [4.0, 8.0, 16.0, 32.0, 64.0];

#[derive(Clone, Copy)]
struct Query {
    center: [f32; 3],
    r: f32,
}

/// A query read directly from a `mbm-extract` query trace file, still tagged with `lanes`: the
/// width `L` of the original `collides_balls::<L>` call this query was part of (see
/// `mbm_extract::RecordedQuery::lanes`). `lanes == 1` means this was an individually-issued
/// single-configuration validity check; `lanes == L > 1` means it was one lane of an `L`-wide
/// batch issued together for a raked motion-segment validity check.
#[derive(Clone, Copy)]
struct RawQuery {
    center: [f32; 3],
    r: f32,
    lanes: u8,
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
/// followed by that many `(x, y, z, r, collided, lanes)` little-endian records.
fn read_queries(path: impl AsRef<Path>) -> Result<Vec<RawQuery>, Box<dyn Error>> {
    let mut r = BufReader::new(File::open(path)?);
    let mut count_buf = [0u8; 8];
    r.read_exact(&mut count_buf)?;
    let count = u64::from_le_bytes(count_buf) as usize;

    let mut queries = Vec::with_capacity(count);
    let mut buf = [0u8; 18];
    for _ in 0..count {
        r.read_exact(&mut buf)?;
        queries.push(RawQuery {
            center: [
                f32::from_le_bytes(buf[0..4].try_into().unwrap()),
                f32::from_le_bytes(buf[4..8].try_into().unwrap()),
                f32::from_le_bytes(buf[8..12].try_into().unwrap()),
            ],
            r: f32::from_le_bytes(buf[12..16].try_into().unwrap()),
            lanes: buf[17],
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

/// Split a raw query trace (in original planner-issue order) back into the two kinds of queries
/// the planner actually issues: individually-issued single-configuration checks (`lanes == 1`)
/// and `L`-wide SIMD batches issued together as one `collides_balls::<L>` call for raked
/// motion-segment validity checks (`lanes == L`).
fn split_batches<const L: usize>(raw: &[RawQuery]) -> (Vec<Query>, Vec<[Query; L]>) {
    let mut scalar = Vec::new();
    let mut batches = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        let lanes = raw[i].lanes as usize;
        if lanes == 1 {
            scalar.push(Query {
                center: raw[i].center,
                r: raw[i].r,
            });
            i += 1;
        } else if lanes == L
            && i + L <= raw.len()
            && raw[i..i + L].iter().all(|q| q.lanes as usize == L)
        {
            let batch = std::array::from_fn(|j| Query {
                center: raw[i + j].center,
                r: raw[i + j].r,
            });
            batches.push(batch);
            i += L;
        } else {
            // A lane width this benchmark doesn't model (or a truncated trailing batch) - drop
            // it rather than risk misgrouping it with unrelated neighbors.
            i += 1;
        }
    }
    (scalar, batches)
}

/// Deterministically subsample `items` down to `size` (or return them unchanged if already at
/// most `size`), seeded by `seed` so runs are reproducible.
fn subsample<T: Clone>(items: Vec<T>, size: usize, seed: u64) -> Vec<T> {
    if items.len() <= size {
        return items;
    }
    let mut rng = SmallRng::seed_from_u64(seed);
    let idx = rand::seq::index::sample(&mut rng, items.len(), size);
    idx.into_iter().map(|i| items[i].clone()).collect()
}

/// Partition the indices of `items` by `pred`, returning `(matching, not_matching)`.
fn partition_indices<T>(items: &[T], mut pred: impl FnMut(&T) -> bool) -> (Vec<usize>, Vec<usize>) {
    let mut yes = Vec::new();
    let mut no = Vec::new();
    for (i, item) in items.iter().enumerate() {
        if pred(item) {
            yes.push(i);
        } else {
            no.push(i);
        }
    }
    (yes, no)
}

/// Convert one true SIMD batch (`L` queries issued together as a single `collides_balls::<L>`
/// call during planning) into `capt`/`mvtable`'s `collides_simd` SIMD input format.
fn batch_to_simd<const L: usize>(batch: &[Query; L]) -> ([Simd<f32, L>; 3], Simd<f32, L>) {
    let centers: [Simd<f32, L>; 3] = std::array::from_fn(|k| {
        Simd::from_array(std::array::from_fn(|lane| batch[lane].center[k]))
    });
    let radii = Simd::from_array(std::array::from_fn(|lane| batch[lane].r));
    (centers, radii)
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
    <Simd<f32, L> as SimdPartialEq>::Mask: Copy,
{
    let tic = Instant::now();
    for (centers, radii) in batches {
        black_box(structure.collides_simd(centers, *radii));
    }
    tic.elapsed()
}

/// Split `items` into `(all, matching, not_matching)` reference traces (using the precomputed
/// `matching`/`not_matching` index sets), skip empty traces, and call `record` with each
/// non-empty `(trace_name, trace)` pair.
fn for_each_trace<'q, T>(
    items: &'q [T],
    matching: &[usize],
    not_matching: &[usize],
    mut record: impl FnMut(&str, &[&'q T]) -> Result<(), Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    let all: Vec<&T> = items.iter().collect();
    let matching: Vec<&T> = matching.iter().map(|&i| &items[i]).collect();
    let not_matching: Vec<&T> = not_matching.iter().map(|&i| &items[i]).collect();
    for (trace_name, trace) in [
        ("all", all),
        ("colliding", matching),
        ("non_colliding", not_matching),
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
    batches: &[[Query; SIMD_L]],
    colliding: &[usize],
    non_colliding: &[usize],
) -> Result<(), Box<dyn Error>> {
    for_each_trace(batches, colliding, non_colliding, |trace_name, trace| {
        let simd_batches: Vec<_> = trace.iter().map(|b| batch_to_simd::<SIMD_L>(b)).collect();
        let n_simd_queries = simd_batches.len() * SIMD_L;
        let elapsed_ns = time_simd_queries(structure, &simd_batches).as_secs_f64() * 1e9;
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
        let raw_queries = read_queries(raw_dir.join(format!("{prefix}_queries.bin")))?;
        let r_max = raw_queries.iter().fold(0.0f32, |m, q| m.max(q.r));
        let r_min = raw_queries.iter().fold(f32::INFINITY, |m, q| m.min(q.r));
        let r_range = (r_min, r_max);

        // Regroup the flat, temporally-ordered trace back into the two kinds of queries the
        // planner actually issued before subsampling.
        let (all_scalar, all_batches) = split_batches::<SIMD_L>(&raw_queries);

        // Subsampled once per workload (not per filter/radius), so every (filter, radius)
        // combination below is compared on the exact same query set.
        let seed = prefix
            .bytes()
            .fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(b as u64));
        let scalar_queries = subsample(all_scalar, MAX_QUERIES, seed);
        let simd_batches = subsample(all_batches, MAX_SIMD_BATCHES, seed.wrapping_add(1));

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
                // with `mvtable` itself. A batch counts as "colliding" if any of its lanes does,
                // mirroring `collides_simd`'s any-of-batch semantics.
                let oracle = mvtable::Mvt::<3, f32>::new(&points, r_max);
                let (colliding, non_colliding) = partition_indices(&scalar_queries, |q| {
                    Structure::collides(&oracle, &q.center, q.r)
                });
                let (batch_colliding, batch_non_colliding) =
                    partition_indices(&simd_batches, |batch| {
                        batch
                            .iter()
                            .any(|q| Structure::collides(&oracle, &q.center, q.r))
                    });

                println!(
                    "{label} [{filter_name} x{scale}] @ {n_points} points (from \
                     {}): {} scalar queries ({} colliding, {} non-colliding), {} SIMD batches \
                     ({} colliding, {} non-colliding)",
                    full_points.len(),
                    scalar_queries.len(),
                    colliding.len(),
                    non_colliding.len(),
                    simd_batches.len(),
                    batch_colliding.len(),
                    batch_non_colliding.len(),
                );

                let n_queries = scalar_queries.len() + simd_batches.len() * SIMD_L;

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
                    n_queries,
                    tic.elapsed().as_secs_f64() * 1e9,
                )?;
                write_memory_row(&mut out, ctx, mvt.memory_used())?;
                bench_scalar(
                    &mut out,
                    ctx,
                    &mvt,
                    &scalar_queries,
                    &colliding,
                    &non_colliding,
                )?;
                if !simd_batches.is_empty() {
                    bench_simd(
                        &mut out,
                        ctx,
                        &mvt,
                        &simd_batches,
                        &batch_colliding,
                        &batch_non_colliding,
                    )?;
                }

                let ctx = RowContext {
                    structure: "mvtable_mutable",
                    ..ctx
                };
                let tic = Instant::now();
                let mvt_mutable = mvtable::MutableMvt::<3, f32>::new(&points, r_max);
                write_construction_row(
                    &mut out,
                    ctx,
                    n_queries,
                    tic.elapsed().as_secs_f64() * 1e9,
                )?;
                write_memory_row(&mut out, ctx, mvt_mutable.memory_used())?;
                bench_scalar(
                    &mut out,
                    ctx,
                    &mvt_mutable,
                    &scalar_queries,
                    &colliding,
                    &non_colliding,
                )?;
                if !simd_batches.is_empty() {
                    bench_simd(
                        &mut out,
                        ctx,
                        &mvt_mutable,
                        &simd_batches,
                        &batch_colliding,
                        &batch_non_colliding,
                    )?;
                }

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
                    n_queries,
                    tic.elapsed().as_secs_f64() * 1e9,
                )?;
                write_memory_row(&mut out, ctx, capt.memory_used())?;
                bench_scalar(
                    &mut out,
                    ctx,
                    &capt,
                    &scalar_queries,
                    &colliding,
                    &non_colliding,
                )?;
                if !simd_batches.is_empty() {
                    bench_simd(
                        &mut out,
                        ctx,
                        &capt,
                        &simd_batches,
                        &batch_colliding,
                        &batch_non_colliding,
                    )?;
                }

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
                    n_queries,
                    tic.elapsed().as_secs_f64() * 1e9,
                )?;
                bench_scalar(
                    &mut out,
                    ctx,
                    &kdt,
                    &scalar_queries,
                    &colliding,
                    &non_colliding,
                )?;

                out.flush()?;
            }
        }
    }

    println!("wrote {}", out_path.display());

    Ok(())
}
