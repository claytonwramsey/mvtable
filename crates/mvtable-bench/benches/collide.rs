//! Construction- and query-time comparison of `mvtable` against `capt` and `kiddo`'s immutable
//! k-d tree, across a range of point cloud sizes.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mvtable_bench::{Structure, uniform_cloud};
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

/// Queries centered exactly on existing points, so every query is guaranteed to collide. This
/// is the case a center-out voxel search order should help most: the colliding point is always
/// in the voxel containing the query center.
fn colliding_queries(points: &[[f32; 3]], rng: &mut SmallRng, n: usize) -> Vec<([f32; 3], f32)> {
    (0..n)
        .map(|_| (points[rng.random_range(0..points.len())], R_MAX / 2.0))
        .collect()
}

/// Queries with a zero radius at fresh, independently-drawn random centers. Since the point
/// cloud and the query centers are both continuous random draws, the probability of an exact
/// coincidence is zero, so this trace never collides: the search block must always be scanned
/// in full, regardless of order.
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
            b.iter(|| black_box(S::build(points, R_MAX)));
        });
    }
}

fn construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction");
    bench_construction::<mvtable::Mvt<3, f32>>(&mut group);
    bench_construction::<capt::Capt<3, f32, u32>>(&mut group);
    bench_construction::<kiddo::ImmutableKdTree<f32, 3>>(&mut group);
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
        let structure = S::build(&points, R_MAX);

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
        bench_query::<capt::Capt<3, f32, u32>>(&mut group, trace_name, trace_of);
        bench_query::<kiddo::ImmutableKdTree<f32, 3>>(&mut group, trace_name, trace_of);
    }
    group.finish();
}

criterion_group!(benches, construction, query);
criterion_main!(benches);
