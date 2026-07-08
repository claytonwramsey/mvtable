//! Exhaustive correctness tests for `mvtable`, cross-checked against `capt`, `kiddo`'s immutable
//! k-d tree, and a brute-force reference implementation.
//!
//! Each test builds every [`Structure`] over the same point cloud, then runs a large batch of
//! queries (a dense deterministic grid sweep, exact-on-point queries, and/or targeted edge
//! cases) through all three structures, asserting that every one of them agrees with the
//! brute-force oracle. Disagreement panics with the full query context needed to reproduce it.

use mvtable_bench::{
    Structure, brute_force_collides, clustered_cloud, flatten_axis, lattice_cloud, query_grid,
    uniform_cloud,
};
use rand::{SeedableRng, rngs::SmallRng};

/// Build every structure over `points`/`r_max`, then check every `(center, radius)` in `queries`
/// against the brute-force oracle, panicking with full context on any disagreement.
fn check_all<const K: usize>(points: &[[f32; K]], r_max: f32, queries: &[([f32; K], f32)]) {
    let mvt = mvtable::Mvt::<K, f32>::build(points, r_max);
    let capt = capt::Capt::<K, f32, u32>::build(points, r_max);
    let kdt = kiddo::ImmutableKdTree::<f32, K>::build(points, r_max);

    for &(center, radius) in queries {
        let expected = brute_force_collides(points, &center, radius);

        for (name, actual) in [
            (mvtable::Mvt::<K, f32>::NAME, mvt.collides(&center, radius)),
            (
                capt::Capt::<K, f32, u32>::NAME,
                capt.collides(&center, radius),
            ),
            (
                kiddo::ImmutableKdTree::<f32, K>::NAME,
                kdt.collides(&center, radius),
            ),
        ] {
            assert_eq!(
                actual,
                expected,
                "{name} disagreed with brute force: K={K}, {} points, r_max={r_max}, \
                 center={center:?}, radius={radius}",
                points.len(),
            );
        }
    }
}

/// Combine a dense deterministic grid sweep with exact-on-point queries, covering both
/// systematic coverage of the query space and the boundary cases that land precisely on a
/// point (the sharpest possible test of the "does this point collide with itself" case).
fn sweep_queries<const K: usize>(
    points: &[[f32; K]],
    half_width: f32,
    grid_resolution: usize,
    radii: &[f32],
) -> Vec<([f32; K], f32)> {
    let mut queries = Vec::new();
    for center in query_grid::<K>(half_width, grid_resolution) {
        for &radius in radii {
            queries.push((center, radius));
        }
    }
    for &p in points {
        queries.push((p, 0.0));
        queries.push((p, 1e-4));
    }
    queries
}

#[test]
fn exhaustive_uniform_1d() {
    let mut rng = SmallRng::seed_from_u64(1);
    let points = uniform_cloud::<_, 1>(&mut rng, 100, 5.0);
    let r_max = 0.5;
    let queries = sweep_queries(&points, 6.0, 400, &[0.0, 0.05, 0.2, 0.5]);
    check_all(&points, r_max, &queries);
}

#[test]
fn exhaustive_uniform_2d() {
    let mut rng = SmallRng::seed_from_u64(2);
    let points = uniform_cloud::<_, 2>(&mut rng, 200, 5.0);
    let r_max = 0.5;
    let queries = sweep_queries(&points, 6.0, 40, &[0.0, 0.05, 0.2, 0.5]);
    check_all(&points, r_max, &queries);
}

#[test]
fn exhaustive_uniform_3d() {
    let mut rng = SmallRng::seed_from_u64(3);
    let points = uniform_cloud::<_, 3>(&mut rng, 300, 5.0);
    let r_max = 0.6;
    let queries = sweep_queries(&points, 6.0, 12, &[0.0, 0.1, 0.3, 0.6]);
    check_all(&points, r_max, &queries);
}

#[test]
fn exhaustive_clustered_3d() {
    let mut rng = SmallRng::seed_from_u64(4);
    let points = clustered_cloud::<_, 3>(&mut rng, 400, 6, 5.0, 0.3);
    let r_max = 0.4;
    let queries = sweep_queries(&points, 6.0, 12, &[0.0, 0.05, 0.2, 0.4]);
    check_all(&points, r_max, &queries);
}

#[test]
fn exhaustive_degenerate_axis() {
    // a planar cloud embedded in 3D space: every point shares the same z coordinate, so the
    // grid has zero true extent along that axis.
    let mut rng = SmallRng::seed_from_u64(5);
    let points = flatten_axis(uniform_cloud::<_, 3>(&mut rng, 250, 5.0), 2, 1.25);
    let r_max = 0.5;
    let queries = sweep_queries(&points, 6.0, 12, &[0.0, 0.1, 0.5, 1.0]);
    check_all(&points, r_max, &queries);
}

#[test]
fn exhaustive_non_cubic_workspace() {
    // a long, thin cloud: 100 units wide along x, but only 2 units tall along y, forcing a
    // heavily non-cubic grid.
    let points: Vec<[f32; 2]> = (0..400)
        .map(|i| [i as f32 * 0.25, (i % 5) as f32 * 0.5])
        .collect();
    let r_max = 0.3;
    let queries = sweep_queries(&points, 110.0, 60, &[0.0, 0.05, 0.15, 0.3]);
    check_all(&points, r_max, &queries);
}

#[test]
fn exhaustive_lattice_boundary() {
    // points sitting exactly on a regular lattice, with `r_max` an exact multiple of the
    // lattice pitch: query centers are placed on lattice points, cell midpoints, and cell
    // corners, deliberately targeting the floating-point voxel-boundary edge cases discussed
    // for `Mvt::collides`.
    const PITCH: f32 = 0.25;
    let points = lattice_cloud::<3>(8, PITCH);
    let r_max = PITCH;

    let mut queries = Vec::new();
    for i in 0..9 {
        for j in 0..9 {
            for k in 0..9 {
                let corner = [i as f32 * PITCH, j as f32 * PITCH, k as f32 * PITCH];
                let mid = [
                    corner[0] + PITCH / 2.0,
                    corner[1] + PITCH / 2.0,
                    corner[2] + PITCH / 2.0,
                ];
                for &radius in &[0.0, PITCH / 4.0, PITCH / 2.0, PITCH, PITCH * 1.5] {
                    queries.push((corner, radius));
                    queries.push((mid, radius));
                }
            }
        }
    }
    check_all(&points, r_max, &queries);
}

#[test]
fn exhaustive_4d() {
    let mut rng = SmallRng::seed_from_u64(6);
    let points = uniform_cloud::<_, 4>(&mut rng, 300, 3.0);
    let r_max = 0.5;
    let queries = sweep_queries::<4>(&points, 3.5, 5, &[0.0, 0.1, 0.3, 0.5]);
    check_all(&points, r_max, &queries);
}

#[test]
fn empty_cloud() {
    let points: Vec<[f32; 3]> = vec![];
    let queries = vec![([0.0, 0.0, 0.0], 0.0), ([1.0, 2.0, 3.0], 100.0)];
    check_all(&points, 1.0, &queries);
}

#[test]
fn single_point() {
    let points = vec![[1.0, 1.0, 1.0]];
    let queries = sweep_queries(&points, 3.0, 10, &[0.0, 0.01, 0.1, 1.0]);
    check_all(&points, 1.0, &queries);
}

#[test]
fn duplicate_points() {
    let points = vec![[0.0, 0.0]; 50];
    let queries = sweep_queries(&points, 2.0, 20, &[0.0, 0.01, 0.5]);
    check_all(&points, 0.5, &queries);
}

#[test]
fn two_points_far_apart() {
    let points = vec![[-100.0, -100.0], [100.0, 100.0]];
    let queries = sweep_queries(&points, 150.0, 30, &[0.0, 0.5, 5.0]);
    check_all(&points, 5.0, &queries);
}

/// `mvtable`'s search-block computation was shown analytically to stay correct even for query
/// radii larger than the `r_max` used at construction (unlike a fixed-neighbor-block scheme).
/// `capt`'s docs explicitly disclaim correctness outside its constructed radius range, so it is
/// deliberately excluded from this comparison.
#[test]
fn radius_beyond_r_max() {
    let mut rng = SmallRng::seed_from_u64(7);
    let points = uniform_cloud::<_, 3>(&mut rng, 200, 5.0);
    let r_max = 0.1;

    let mvt = mvtable::Mvt::<3, f32>::build(&points, r_max);
    let kdt = kiddo::ImmutableKdTree::<f32, 3>::build(&points, r_max);

    for center in query_grid::<3>(6.0, 10) {
        for &radius in &[0.5, 1.0, 3.0] {
            let expected = brute_force_collides(&points, &center, radius);
            assert_eq!(
                mvt.collides(&center, radius),
                expected,
                "mvtable disagreed with brute force for radius > r_max: center={center:?}, \
                 radius={radius}"
            );
            assert_eq!(
                kdt.collides(&center, radius),
                expected,
                "kiddo disagreed with brute force: center={center:?}, radius={radius}"
            );
        }
    }
}

/// `mvtable::Mvt::with_point_radius` isn't part of the shared [`Structure`] interface (`capt` is
/// the only other structure that models point radius, and `kiddo` cannot), so it's checked here
/// directly against a brute-force oracle with the point radius folded into the query radius.
#[test]
fn point_radius() {
    let mut rng = SmallRng::seed_from_u64(8);
    let points = uniform_cloud::<_, 3>(&mut rng, 150, 5.0);
    let r_max = 0.3;
    let r_point = 0.05;

    let mvt = mvtable::Mvt::<3, f32>::with_point_radius(&points, r_max, r_point);
    let capt = capt::Capt::<3, f32, u32>::with_point_radius(&points, (0.0, r_max), r_point, 1);

    for center in query_grid::<3>(6.0, 10) {
        for &radius in &[0.0, 0.05, 0.15, 0.3] {
            let expected = brute_force_collides(&points, &center, radius + r_point);
            assert_eq!(
                mvt.collides(&center, radius),
                expected,
                "mvtable with_point_radius disagreed with brute force: center={center:?}, \
                 radius={radius}"
            );
            assert_eq!(
                capt.collides(&center, radius),
                expected,
                "capt with_point_radius disagreed with brute force: center={center:?}, \
                 radius={radius}"
            );
        }
    }
}
