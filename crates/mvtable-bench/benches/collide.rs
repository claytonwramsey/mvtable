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

fn make_queries(rng: &mut SmallRng, n: usize) -> Vec<([f32; 3], f32)> {
    (0..n)
        .map(|_| {
            (
                std::array::from_fn(|_| rng.random_range(-HALF_WIDTH..HALF_WIDTH)),
                rng.random_range(0.0..R_MAX),
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
) {
    let mut rng = SmallRng::seed_from_u64(1);
    for &n in &SIZES {
        let points: Vec<[f32; 3]> = uniform_cloud(&mut rng, n, HALF_WIDTH);
        let queries = make_queries(&mut rng, N_QUERIES);
        let structure = S::build(&points, R_MAX);

        group.bench_with_input(BenchmarkId::new(S::NAME, n), &queries, |b, queries| {
            b.iter(|| {
                for (center, radius) in queries {
                    black_box(structure.collides(center, *radius));
                }
            });
        });
    }
}

fn query(c: &mut Criterion) {
    let mut group = c.benchmark_group("query");
    bench_query::<mvtable::Mvt<3, f32>>(&mut group);
    bench_query::<capt::Capt<3, f32, u32>>(&mut group);
    bench_query::<kiddo::ImmutableKdTree<f32, 3>>(&mut group);
    group.finish();
}

criterion_group!(benches, construction, query);
criterion_main!(benches);
