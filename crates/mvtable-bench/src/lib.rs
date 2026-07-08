//! Shared utilities for benchmarking and correctness-testing `mvtable` against `capt` and
//! `kiddo`'s immutable k-d tree.
//!
//! The [`Structure`] trait gives all three collision-checking structures a common
//! construct-then-query interface, so that both `benches/collide.rs` and `tests/correctness.rs`
//! can be written once and run against every structure.

use kiddo::SquaredEuclidean;
use rand::{Rng, RngExt};

/// A minimal common interface over the collision-checking structures being compared.
pub trait Structure<const K: usize>: Sized {
    /// A short, human-readable name for this structure, used in benchmark/test output.
    const NAME: &'static str;

    /// Build a new instance containing `points`, sized for queries with radius up to `r_max`.
    fn build(points: &[[f32; K]], r_max: f32) -> Self;

    /// Determine whether any point in the structure lies within `radius` of `center`.
    fn collides(&self, center: &[f32; K], radius: f32) -> bool;
}

impl<const K: usize> Structure<K> for mvtable::Mvt<K, f32> {
    const NAME: &'static str = "mvtable";

    fn build(points: &[[f32; K]], r_max: f32) -> Self {
        Self::new(points, r_max)
    }

    fn collides(&self, center: &[f32; K], radius: f32) -> bool {
        Self::collides(self, center, radius)
    }
}

impl<const K: usize> Structure<K> for capt::Capt<K, f32, u32> {
    const NAME: &'static str = "capt";

    fn build(points: &[[f32; K]], r_max: f32) -> Self {
        Self::new(points, (0.0, r_max), 1)
    }

    fn collides(&self, center: &[f32; K], radius: f32) -> bool {
        Self::collides(self, center, radius)
    }
}

impl<const K: usize> Structure<K> for kiddo::ImmutableKdTree<f32, K> {
    const NAME: &'static str = "kiddo";

    fn build(points: &[[f32; K]], _r_max: f32) -> Self {
        Self::new_from_slice(points)
    }

    fn collides(&self, center: &[f32; K], radius: f32) -> bool {
        // an empty tree has no nearest neighbor to query for.
        self.size() != 0
            && !self
                .within_unsorted::<SquaredEuclidean>(center, radius * radius)
                .is_empty()
    }
}

/// Compute the exact answer to a collision query by checking every point in `points`.
///
/// This is the ground truth against which every [`Structure`] implementation is checked.
#[must_use]
pub fn brute_force_collides<const K: usize>(
    points: &[[f32; K]],
    center: &[f32; K],
    radius: f32,
) -> bool {
    let rsq = radius * radius;
    points.iter().any(|p| {
        let mut distsq = 0.0f32;
        for k in 0..K {
            let d = p[k] - center[k];
            distsq += d * d;
        }
        distsq <= rsq
    })
}

/// Generate `n` points drawn uniformly at random from the axis-aligned box
/// `[-half_width, half_width]^K`.
pub fn uniform_cloud<R: Rng + ?Sized, const K: usize>(
    rng: &mut R,
    n: usize,
    half_width: f32,
) -> Vec<[f32; K]> {
    (0..n)
        .map(|_| std::array::from_fn(|_| rng.random_range(-half_width..half_width)))
        .collect()
}

/// Generate `n` points drawn from `n_clusters` tight clusters within
/// `[-half_width, half_width]^K`, to model non-uniform, structured point clouds (e.g. clumps of
/// obstacle points rather than a uniform gas).
pub fn clustered_cloud<R: Rng + ?Sized, const K: usize>(
    rng: &mut R,
    n: usize,
    n_clusters: usize,
    half_width: f32,
    cluster_radius: f32,
) -> Vec<[f32; K]> {
    let n_clusters = n_clusters.max(1);
    let centers: Vec<[f32; K]> = (0..n_clusters)
        .map(|_| std::array::from_fn(|_| rng.random_range(-half_width..half_width)))
        .collect();
    (0..n)
        .map(|_| {
            let c = centers[rng.random_range(0..n_clusters)];
            std::array::from_fn(|k| c[k] + rng.random_range(-cluster_radius..cluster_radius))
        })
        .collect()
}

/// Generate points on a regular lattice with spacing `pitch`, `n_per_axis` points along each
/// axis (`n_per_axis.pow(K)` points in total), anchored at the origin.
///
/// Useful for stressing floating-point voxel-boundary edge cases: every point (and many query
/// centers, if generated at multiples or half-multiples of `pitch`) lands exactly on a cell
/// boundary of a grid sized with a matching `r_max`.
#[must_use]
pub fn lattice_cloud<const K: usize>(n_per_axis: usize, pitch: f32) -> Vec<[f32; K]> {
    let total = n_per_axis.pow(u32::try_from(K).expect("K should fit in a u32"));
    (0..total)
        .map(|mut idx| {
            std::array::from_fn(|_| {
                let coord = (idx % n_per_axis) as f32 * pitch;
                idx /= n_per_axis;
                coord
            })
        })
        .collect()
}

/// Overwrite axis `flat_axis` of every point in `points` with `value`, producing a degenerate
/// (lower-dimensional) cloud for testing how axes with zero extent are handled.
#[must_use]
pub fn flatten_axis<const K: usize>(
    mut points: Vec<[f32; K]>,
    flat_axis: usize,
    value: f32,
) -> Vec<[f32; K]> {
    for p in &mut points {
        p[flat_axis] = value;
    }
    points
}

/// Generate a dense, deterministic grid of `resolution.pow(K)` query centers covering
/// `[-half_width, half_width]^K`, for exhaustive-style coverage of a continuous query space that
/// random sampling alone might miss.
#[must_use]
pub fn query_grid<const K: usize>(half_width: f32, resolution: usize) -> Vec<[f32; K]> {
    let total = resolution.pow(u32::try_from(K).expect("K should fit in a u32"));
    let step = 2.0 * half_width / resolution as f32;
    (0..total)
        .map(|mut idx| {
            std::array::from_fn(|_| {
                let cell = idx % resolution;
                idx /= resolution;
                -half_width + step * (cell as f32 + 0.5)
            })
        })
        .collect()
}
