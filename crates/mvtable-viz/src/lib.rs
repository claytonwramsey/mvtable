//! Shared scene generation for `mvtable-viz`'s figures, kept in one place so every figure
//! (`grid_construction`, `mvt_query`, `kdtree_query`, `capt_query`) visualizes the exact same
//! synthetic point cloud and query spheres, making them directly comparable.

use std::f32::consts::TAU;

use rand::{RngExt, SeedableRng, rngs::SmallRng, seq::SliceRandom};

pub mod tree;

/// The dimension used by every figure in this crate.
pub const K: usize = 3;

/// A point.
pub type Point = [f32; K];

/// An axis-aligned bounding box, with the same clamped-distance semantics as `mvtable`'s and
/// `capt`'s internal `Aabb` types.
#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub lo: Point,
    pub hi: Point,
}

impl Aabb {
    /// The bounding box over `points`. Panics if `points` is empty.
    #[must_use]
    pub fn of(points: &[Point]) -> Self {
        let mut lo = [f32::INFINITY; K];
        let mut hi = [f32::NEG_INFINITY; K];
        for p in points {
            for k in 0..K {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        Self { lo, hi }
    }

    /// `self`, expanded outward by `margin` on every axis.
    #[must_use]
    pub fn dilate(&self, margin: f32) -> Self {
        Self {
            lo: std::array::from_fn(|k| self.lo[k] - margin),
            hi: std::array::from_fn(|k| self.hi[k] + margin),
        }
    }

    /// The squared distance from `p` to the closest point of this box (zero if `p` is inside).
    #[must_use]
    pub fn closest_distsq_to(&self, p: &Point) -> f32 {
        (0..K)
            .map(|k| (p[k] - p[k].clamp(self.lo[k], self.hi[k])).powi(2))
            .sum()
    }

    /// The size of this box along each axis.
    #[must_use]
    pub fn size(&self) -> Point {
        std::array::from_fn(|k| self.hi[k] - self.lo[k])
    }
}

/// Squared distance between two points.
#[must_use]
pub fn distsq(a: &Point, b: &Point) -> f32 {
    (0..K).map(|k| (a[k] - b[k]).powi(2)).sum()
}

/// A linear-scan collision check, used as ground truth to validate the tree-search figures'
/// hand-rolled traversals against.
#[must_use]
pub fn brute_force_collides(points: &[Point], center: Point, radius: f32) -> bool {
    let rsq = radius * radius;
    points.iter().any(|p| distsq(p, &center) <= rsq)
}

/// Sample `n` points on the surface of a sphere, with a little radial jitter so it reads as a
/// noisy sensor scan rather than a mathematically perfect shell.
fn sample_sphere_surface(
    rng: &mut SmallRng,
    center: Point,
    radius: f32,
    n: usize,
    out: &mut Vec<Point>,
) {
    for _ in 0..n {
        let z: f32 = rng.random_range(-1.0..1.0);
        let theta: f32 = rng.random_range(0.0..TAU);
        let r_xy = (1.0 - z * z).sqrt();
        let jitter = radius * (1.0 + rng.random_range(-0.03..0.03));
        out.push([
            center[0] + jitter * r_xy * theta.cos(),
            center[1] + jitter * r_xy * theta.sin(),
            center[2] + jitter * z,
        ]);
    }
}

/// The shared point cloud used by every figure: two noisy sphere-surface scans, in a fixed,
/// seeded (and therefore reproducible) shuffled order.
#[must_use]
pub fn point_cloud() -> Vec<Point> {
    let mut rng = SmallRng::seed_from_u64(0);
    let mut points = Vec::new();
    sample_sphere_surface(&mut rng, [-1.4, 0.0, 0.0], 1.0, 260, &mut points);
    sample_sphere_surface(&mut rng, [1.1, 0.9, -0.4], 0.65, 170, &mut points);
    points.shuffle(&mut rng);
    points
}

/// How many cells (MVT voxels, or roughly, tree leaves) the longest workspace axis should be
/// carved into, shared by every figure so their granularity is comparable.
pub const TARGET_CELLS_PER_AXIS: f32 = 10.0;

/// The maximum query/collision radius shared by every figure, derived from the point cloud's
/// bounding box the same way [`grid_construction`](../bin/grid_construction) sizes its voxels.
#[must_use]
pub fn r_max(aabb: &Aabb) -> f32 {
    let max_extent = aabb.size().into_iter().fold(0.0f32, f32::max);
    max_extent / TARGET_CELLS_PER_AXIS
}

/// The two query spheres used by every "search" figure: one that collides with the cloud, one
/// that doesn't, so both a pruned/miss path and a hit path are visible.
#[must_use]
pub fn queries(r_max: f32) -> [(&'static str, Point, f32); 2] {
    [
        ("miss", [0.0, 0.0, 1.6], r_max * 0.9),
        ("hit", [-1.4, 0.0, 1.0], r_max * 0.9),
    ]
}

/// Mirrors `mvtable`'s internal grid-sizing formula (`crates/mvtable/src/grid.rs::size_grid`), so
/// figures can compute voxel cell boundaries in world space without needing access to
/// `MutableMvt`'s private table layout: an MVT's voxels are cells of a uniform grid of width
/// `cell_wd` anchored at `lo`.
#[must_use]
pub fn grid_params(aabb: &Aabb, cell_wd: f32) -> ([usize; K], Point) {
    let mut grid_width = [0usize; K];
    let mut scale = [0.0f32; K];
    for k in 0..K {
        let extent = aabb.hi[k] - aabb.lo[k];
        let extent = if extent > 0.0 { extent } else { cell_wd };
        let gw = usize::max(1, (extent / cell_wd) as usize);
        grid_width[k] = gw;
        scale[k] = gw as f32 / extent;
    }
    (grid_width, scale)
}

/// Same clamped-floor-division mapping as `mvtable`'s `grid::point_to_grid_coords`.
#[must_use]
pub fn point_to_coords(p: &Point, lo: Point, scale: Point, grid_width: [usize; K]) -> [usize; K] {
    std::array::from_fn(|k| (((p[k] - lo[k]) * scale[k]) as usize).min(grid_width[k] - 1))
}

/// Convert a `[f32; 3]` point into the tuple form Rerun's archetypes expect.
#[must_use]
pub fn v3(p: Point) -> (f32, f32, f32) {
    (p[0], p[1], p[2])
}
