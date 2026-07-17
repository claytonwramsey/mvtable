//! Construction- and query-time comparison of `mvtable` against `capt`, `kiddo`'s immutable
//! k-d tree, and `mvt_cpp` (the vendored C++ reference implementation, see
//! `crates/mvt-cpp/vendor/README.md`).
#![feature(portable_simd)]

use std::{hint::black_box, simd::Simd};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mvtable_bench::{SimdStructure, Structure, uniform_cloud};
use rand::{RngExt, SeedableRng, rngs::SmallRng};

const R_MAX: f32 = 0.05;
const HALF_WIDTH: f32 = 5.0;
const SIZES: [usize; 3] = [1_000, 10_000, 100_000];
const N_QUERIES: usize = 2_000;

/// A mix of colliding and non-colliding queries, uniformly distributed over the whole workspace.
fn mixed_queries(rng: &mut SmallRng, n: usize) -> Vec<([f32; 3], f32)> {
    (0..n)
        .map(|_| {
            (
                std::array::from_fn(|_| rng.random_range(-HALF_WIDTH..HALF_WIDTH)),
                rng.random_range(0.0..R_MAX),
            )
        })
        .collect()
}

/// Queries centered exactly on existing points, so every query is guaranteed to collide.
fn colliding_queries(points: &[[f32; 3]], rng: &mut SmallRng, n: usize) -> Vec<([f32; 3], f32)> {
    (0..n)
        .map(|_| (points[rng.random_range(0..points.len())], R_MAX / 2.0))
        .collect()
}

/// Queries with a zero radius at fresh, independently-drawn random centers. Since the point
/// cloud and the query centers are both continuous random draws, the probability of an exact
/// coincidence is zero.
fn non_colliding_queries(rng: &mut SmallRng, n: usize) -> Vec<([f32; 3], f32)> {
    (0..n)
        .map(|_| {
            (
                std::array::from_fn(|_| rng.random_range(-HALF_WIDTH..HALF_WIDTH)),
                0.0,
            )
        })
        .collect()
}

fn bench_construction<S: Structure<3>>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
) {
    let mut rng = SmallRng::seed_from_u64(0);
    for &n in &SIZES {
        let points: Vec<[f32; 3]> = uniform_cloud(&mut rng, n, HALF_WIDTH);
        group.bench_with_input(BenchmarkId::new(S::NAME, n), &points, |b, points| {
            b.iter(|| black_box(S::build(points, (0.0, R_MAX))));
        });
    }
}

/// Like [`bench_construction`], but for `mvt_cpp`: not generic over `S: Structure<3>` because it
/// needs to skip (rather than crash on) any `n` that would overflow the vendored implementation's
/// fixed-capacity pools - see `mvt_cpp::Overflow`'s doc comment. Probes with one `try_new` call
/// (dropped immediately) rather than the panicking `Structure::build` used inside the timed loop,
/// since `criterion`'s `b.iter()` closure has no way to skip a benchmark mid-run.
fn bench_construction_mvt_cpp(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    let mut rng = SmallRng::seed_from_u64(0);
    for &n in &SIZES {
        let points: Vec<[f32; 3]> = uniform_cloud(&mut rng, n, HALF_WIDTH);
        if mvt_cpp::MvtCpp::try_new(&points, (0.0, R_MAX)).is_err() {
            eprintln!("skipping mvt_cpp construction bench at n={n}: would overflow its fixed-capacity pools");
            continue;
        }
        let id = BenchmarkId::new(<mvt_cpp::MvtCpp as Structure<3>>::NAME, n);
        group.bench_with_input(id, &points, |b, points| {
            b.iter(|| black_box(<mvt_cpp::MvtCpp as Structure<3>>::build(points, (0.0, R_MAX))));
        });
    }
}

fn construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction");
    bench_construction::<mvtable::Mvt<3, f32>>(&mut group);
    bench_construction::<mvtable::MutableMvt<3, f32>>(&mut group);
    bench_construction::<capt::Capt<3, f32, u32>>(&mut group);
    bench_construction::<kiddo::ImmutableKdTree<f32, 3>>(&mut group);
    bench_construction_mvt_cpp(&mut group);
    group.finish();
}

fn bench_query<S: Structure<3>>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    trace_name: &str,
    trace_of: impl Fn(&[[f32; 3]], &mut SmallRng) -> Vec<([f32; 3], f32)>,
) {
    let mut rng = SmallRng::seed_from_u64(1);
    for &n in &SIZES {
        let points: Vec<[f32; 3]> = uniform_cloud(&mut rng, n, HALF_WIDTH);
        let queries = trace_of(&points, &mut rng);
        let structure = S::build(&points, (0.0, R_MAX));

        let id = BenchmarkId::new(format!("{}/{trace_name}", S::NAME), n);
        group.bench_with_input(id, &queries, |b, queries| {
            b.iter(|| {
                for (center, radius) in queries {
                    black_box(structure.collides(center, *radius));
                }
            });
        });
    }
}

/// A named query-generating function, used to label and reuse each query trace across
/// structures.
type TraceFn = fn(&[[f32; 3]], &mut SmallRng) -> Vec<([f32; 3], f32)>;

fn query(c: &mut Criterion) {
    let mut group = c.benchmark_group("query");
    let traces: [(&str, TraceFn); 3] = [
        ("mixed", |_, rng| mixed_queries(rng, N_QUERIES)),
        ("colliding", |points, rng| {
            colliding_queries(points, rng, N_QUERIES)
        }),
        ("non_colliding", |_, rng| {
            non_colliding_queries(rng, N_QUERIES)
        }),
    ];
    for (trace_name, trace_of) in traces {
        bench_query::<mvtable::Mvt<3, f32>>(&mut group, trace_name, trace_of);
        bench_query::<mvtable::MutableMvt<3, f32>>(&mut group, trace_name, trace_of);
        bench_query::<capt::Capt<3, f32, u32>>(&mut group, trace_name, trace_of);
        bench_query::<kiddo::ImmutableKdTree<f32, 3>>(&mut group, trace_name, trace_of);
    }
    group.finish();
}

/// Group `queries` into batches of `L`, converting each batch into a SIMD center/radius vector.
/// Any remainder that doesn't fill a full batch of `L` is dropped.
fn to_simd_batches<const L: usize>(
    queries: &[([f32; 3], f32)],
) -> Vec<([Simd<f32, L>; 3], Simd<f32, L>)> {
    queries
        .chunks_exact(L)
        .map(|chunk| {
            let centers: [Simd<f32, L>; 3] = std::array::from_fn(|k| {
                Simd::from_array(std::array::from_fn(|lane| chunk[lane].0[k]))
            });
            let radii = Simd::from_array(std::array::from_fn(|lane| chunk[lane].1));
            (centers, radii)
        })
        .collect()
}

/// The largest lane width `L` benchmarked; `capt` needs to be constructed with at least this many
/// lanes to be queried with any `L` up to it.
const MAX_L: usize = 8;

/// Benchmark `mvtable`'s and `capt`'s scalar `collides`, once per
/// trace. `mvtable_mutable` is built by inserting every point one at a time (rather than
/// `mvtable`'s single-shot `new`), to also capture the insertion-side cost of `MutableMvt`.
fn bench_scalar_query(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    trace_name: &str,
    trace_of: impl Fn(&[[f32; 3]], &mut SmallRng) -> Vec<([f32; 3], f32)>,
) {
    let mut rng = SmallRng::seed_from_u64(1);
    for &n in &SIZES {
        let points: Vec<[f32; 3]> = uniform_cloud(&mut rng, n, HALF_WIDTH);
        let queries = trace_of(&points, &mut rng);
        let mvt = mvtable::Mvt::<3, f32>::new(&points, R_MAX);
        let mvt_mutable = mvtable::MutableMvt::<3, f32>::new(&points, R_MAX);
        // `n_lanes = 1` matches how `capt` is constructed for the plain scalar `query` group.
        let capt = capt::Capt::<3, f32, u32>::new(&points, (0.0, R_MAX), 1);

        let id = BenchmarkId::new(format!("mvtable_scalar/{trace_name}"), n);
        group.bench_with_input(id, &queries, |b, queries| {
            b.iter(|| {
                for (center, radius) in queries {
                    black_box(mvt.collides(center, *radius));
                }
            });
        });

        let id = BenchmarkId::new(format!("mvtable_mutable_scalar/{trace_name}"), n);
        group.bench_with_input(id, &queries, |b, queries| {
            b.iter(|| {
                for (center, radius) in queries {
                    black_box(mvt_mutable.collides(center, *radius));
                }
            });
        });

        let id = BenchmarkId::new(format!("capt_scalar/{trace_name}"), n);
        group.bench_with_input(id, &queries, |b, queries| {
            b.iter(|| {
                for (center, radius) in queries {
                    black_box(capt.collides(center, *radius));
                }
            });
        });

        let mvt_cpp_instance = match mvt_cpp::MvtCpp::try_new(&points, (0.0, R_MAX)) {
            Ok(instance) => instance,
            Err(mvt_cpp::Overflow) => {
                eprintln!("skipping mvt_cpp scalar query bench at n={n}: would overflow its fixed-capacity pools");
                continue;
            }
        };
        let id = BenchmarkId::new(format!("mvt_cpp_scalar/{trace_name}"), n);
        group.bench_with_input(id, &queries, |b, queries| {
            b.iter(|| {
                for (center, radius) in queries {
                    black_box(mvt_cpp_instance.collides(center, *radius));
                }
            });
        });
    }
}

/// Benchmark inserting `n` points one at a time into a fresh `MutableMvt`, seeded from a single
/// initial point so the workspace roughly matches the final cloud, against building an
/// equivalent `Mvt` in one shot.
fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");
    let mut rng = SmallRng::seed_from_u64(0);
    for &n in &SIZES {
        let points: Vec<[f32; 3]> = uniform_cloud(&mut rng, n, HALF_WIDTH);

        let id = BenchmarkId::new("mvtable", n);
        group.bench_with_input(id, &points, |b, points| {
            b.iter(|| black_box(mvtable::Mvt::<3, f32>::new(points, R_MAX)));
        });

        let id = BenchmarkId::new("mvtable_mutable_insert_one_at_a_time", n);
        group.bench_with_input(id, &points, |b, points| {
            b.iter(|| {
                let mut mvt = mvtable::MutableMvt::<3, f32>::new(&points[..1], R_MAX);
                for p in &points[1..] {
                    mvt.insert(p).unwrap();
                }
                black_box(mvt);
            });
        });
    }
    group.finish();
}

/// Benchmark `mvtable`'s (both `Mvt` and `MutableMvt`) and `capt`'s SIMD-batched `collides_simd`
/// at lane width `L`, using the same point clouds and queries (grouped into batches of `L`) as
/// [`bench_scalar_query`].
fn bench_simd_query<const L: usize>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    trace_name: &str,
    trace_of: impl Fn(&[[f32; 3]], &mut SmallRng) -> Vec<([f32; 3], f32)>,
) {
    let mut rng = SmallRng::seed_from_u64(1);
    for &n in &SIZES {
        let points: Vec<[f32; 3]> = uniform_cloud(&mut rng, n, HALF_WIDTH);
        let queries = trace_of(&points, &mut rng);
        let mvt = mvtable::Mvt::<3, f32>::new(&points, R_MAX);
        let mvt_mutable = mvtable::MutableMvt::<3, f32>::new(&points, R_MAX);
        let capt = capt::Capt::<3, f32, u32>::new(&points, (0.0, R_MAX), MAX_L);
        let batches = to_simd_batches::<L>(&queries);

        let id = BenchmarkId::new(format!("mvtable_simd_l{L}/{trace_name}"), n);
        group.bench_with_input(id, &batches, |b, batches| {
            b.iter(|| {
                for (centers, radii) in batches {
                    black_box(mvt.collides_simd(centers, *radii));
                }
            });
        });

        let id = BenchmarkId::new(format!("mvtable_mutable_simd_l{L}/{trace_name}"), n);
        group.bench_with_input(id, &batches, |b, batches| {
            b.iter(|| {
                for (centers, radii) in batches {
                    black_box(mvt_mutable.collides_simd(centers, *radii));
                }
            });
        });

        let id = BenchmarkId::new(format!("capt_simd_l{L}/{trace_name}"), n);
        group.bench_with_input(id, &batches, |b, batches| {
            b.iter(|| {
                for (centers, radii) in batches {
                    black_box(capt.collides_simd(centers, *radii));
                }
            });
        });
    }
}

/// Like [`bench_simd_query`], but for `mvt_cpp` at its one hardware-fixed lane width
/// (`mvt_cpp::SIMD_WIDTH` - 8 on x86_64/AVX2, matching [`MAX_L`]; 4 on aarch64/NEON, which
/// [`bench_simd_query`]'s `L=4` instantiation happens to already cover) - its vendored SIMD
/// backend doesn't support other lane widths (see `crates/mvt-cpp/vendor/README.md`), and it
/// needs to skip (rather than crash on) any `n` that would overflow its fixed-capacity pools (see
/// `mvt_cpp::Overflow`'s doc comment).
fn bench_simd_query_mvt_cpp(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    trace_name: &str,
    trace_of: impl Fn(&[[f32; 3]], &mut SmallRng) -> Vec<([f32; 3], f32)>,
) {
    let mut rng = SmallRng::seed_from_u64(1);
    for &n in &SIZES {
        let points: Vec<[f32; 3]> = uniform_cloud(&mut rng, n, HALF_WIDTH);
        let mvt_cpp_instance = match mvt_cpp::MvtCpp::try_new(&points, (0.0, R_MAX)) {
            Ok(instance) => instance,
            Err(mvt_cpp::Overflow) => {
                eprintln!("skipping mvt_cpp SIMD query bench at n={n}: would overflow its fixed-capacity pools");
                continue;
            }
        };
        let queries = trace_of(&points, &mut rng);
        let batches = to_simd_batches::<{ mvt_cpp::SIMD_WIDTH }>(&queries);

        let id = BenchmarkId::new(format!("mvt_cpp_simd_l{}/{trace_name}", mvt_cpp::SIMD_WIDTH), n);
        group.bench_with_input(id, &batches, |b, batches| {
            b.iter(|| {
                for (centers, radii) in batches {
                    black_box(SimdStructure::collides_simd(&mvt_cpp_instance, centers, *radii));
                }
            });
        });
    }
}

fn simd_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_query");
    let traces: [(&str, TraceFn); 3] = [
        ("mixed", |_, rng| mixed_queries(rng, N_QUERIES)),
        ("colliding", |points, rng| {
            colliding_queries(points, rng, N_QUERIES)
        }),
        ("non_colliding", |_, rng| {
            non_colliding_queries(rng, N_QUERIES)
        }),
    ];
    for (trace_name, trace_of) in traces {
        bench_scalar_query(&mut group, trace_name, trace_of);
        bench_simd_query::<4>(&mut group, trace_name, trace_of);
        bench_simd_query::<MAX_L>(&mut group, trace_name, trace_of);
        bench_simd_query_mvt_cpp(&mut group, trace_name, trace_of);
    }
    group.finish();
}

criterion_group!(benches, construction, query, simd_query, bench_insert);
criterion_main!(benches);
