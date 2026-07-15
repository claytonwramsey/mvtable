//! Shared utilities for benchmarking and correctness-testing `mvtable` against `capt` and
//! `kiddo`'s immutable k-d tree.
//!
//! The [`Structure`] trait gives all three collision-checking structures a common
//! construct-then-query interface, so that both `benches/collide.rs` and `tests/correctness.rs`
//! can be written once and run against every structure. [`SimdStructure`] additionally covers
//! `mvtable` and `capt`'s SIMD-batched queries (`kiddo` has no SIMD-batched query API).
#![feature(portable_simd)]

use std::simd::{Simd, cmp::SimdPartialEq};

use capt::AxisSimd;
use kiddo::SquaredEuclidean;
use rand::{Rng, RngExt};

pub mod filter;

/// A minimal common interface over the collision-checking structures being compared.
pub trait Structure<const K: usize>: Sized {
    /// A short, human-readable name for this structure, used in benchmark/test output.
    const NAME: &'static str;

    /// Build a new instance containing `points`, sized for queries with radius in `r_range`
    /// (`(r_min, r_max)`). Structures that don't need a lower bound (`mvtable`, `kiddo`) ignore
    /// `r_range.0`.
    fn build(points: &[[f32; K]], r_range: (f32, f32)) -> Self;

    /// Determine whether any point in the structure lies within `radius` of `center`.
    fn collides(&self, center: &[f32; K], radius: f32) -> bool;
}

impl<const K: usize> Structure<K> for mvtable::Mvt<K, f32> {
    const NAME: &'static str = "mvtable";

    fn build(points: &[[f32; K]], r_range: (f32, f32)) -> Self {
        Self::new(points, r_range.1)
    }

    fn collides(&self, center: &[f32; K], radius: f32) -> bool {
        Self::collides(self, center, radius)
    }
}

impl<const K: usize> Structure<K> for mvtable::MutableMvt<K, f32> {
    const NAME: &'static str = "mvtable_mutable";

    fn build(points: &[[f32; K]], r_range: (f32, f32)) -> Self {
        if points.is_empty() {
            // `MutableMvt::new` requires a non-empty point cloud to infer workspace bounds; an
            // empty structure has no points to bound, so any placeholder workspace box works.
            Self::with_workspace([0.0; K], [1.0; K], r_range.1, 0.0)
        } else {
            Self::new(points, r_range.1)
        }
    }

    fn collides(&self, center: &[f32; K], radius: f32) -> bool {
        Self::collides(self, center, radius)
    }
}

impl<const K: usize> Structure<K> for capt::Capt<K, f32, u32> {
    const NAME: &'static str = "capt";

    fn build(points: &[[f32; K]], r_range: (f32, f32)) -> Self {
        Self::new(points, r_range, 1)
    }

    fn collides(&self, center: &[f32; K], radius: f32) -> bool {
        Self::collides(self, center, radius)
    }
}

impl<const K: usize> Structure<K> for kiddo::ImmutableKdTree<f32, K> {
    const NAME: &'static str = "kiddo";

    fn build(points: &[[f32; K]], _r_range: (f32, f32)) -> Self {
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

/// Replicate `std`'s `Vec` amortized-growth policy: given a `Vec` reserved with
/// `Vec::with_capacity(initial_capacity)` and then grown one `push` at a time up to
/// `final_len` elements, return its resulting `capacity()`.
///
/// This is unstable, non-contractual `std`
/// behavior (not guaranteed by the `Vec` API), but has been stable in practice for a long time and
/// was confirmed to match exactly for every case this crate's usage hits.
#[must_use]
fn simulate_vec_growth(initial_capacity: usize, final_len: usize, elem_size: usize) -> usize {
    if final_len == 0 {
        return initial_capacity;
    }
    let mut cap = initial_capacity;
    if cap == 0 {
        // `std`'s `min_non_zero_cap`: 8 for zero-sized-adjacent (1-byte) elements, 1 for large
        // (>1024-byte) elements, 4 otherwise.
        cap = if elem_size == 1 {
            8
        } else if elem_size <= 1024 {
            4
        } else {
            1
        };
    }
    while cap < final_len {
        cap = cap.saturating_mul(2).max(cap + 1);
    }
    cap
}

/// Compute the total memory used (stack + heap) by a `kiddo::ImmutableKdTree<f32, K>`, measured
/// in bytes.
///
/// `kiddo` exposes no such method, and its fields are private, so this is computed analytically
/// from [`kiddo::ImmutableKdTree::size`].
/// This was experimentally validated against a patched kiddo implementation.
#[must_use]
pub fn kiddo_memory_used<const K: usize>(tree: &kiddo::ImmutableKdTree<f32, K>) -> usize {
    /// `kiddo::ImmutableKdTree<A, K>`'s fixed leaf bucket size (its `B` const-generic param).
    const B: usize = 32;

    let item_count = tree.size();
    let leaf_node_count_raw = item_count.div_ceil(B);
    let leaf_node_count = leaf_node_count_raw.max(1);
    let stem_node_count = if leaf_node_count < 2 {
        0
    } else {
        leaf_node_count.next_power_of_two()
    };
    let leaf_extents_len = stem_node_count.max(1);
    let leaf_extents_cap = simulate_vec_growth(
        leaf_node_count_raw,
        leaf_extents_len,
        size_of::<(u32, u32)>(),
    );

    size_of::<kiddo::ImmutableKdTree<f32, K>>()
        + stem_node_count * size_of::<f32>()
        + K * item_count * size_of::<f32>()
        + item_count * size_of::<u64>()
        + leaf_extents_cap * size_of::<(u32, u32)>()
}

/// [`Structure`]s that additionally support SIMD-batched collision queries.
pub trait SimdStructure<const K: usize>: Structure<K> {
    /// Determine whether any point in the structure lies within the corresponding lane of `radii`
    /// of the corresponding lane of `centers`.
    fn collides_simd<const L: usize>(
        &self,
        centers: &[Simd<f32, L>; K],
        radii: Simd<f32, L>,
    ) -> bool
    where
        Simd<f32, L>: AxisSimd<L>,
        <Simd<f32, L> as SimdPartialEq>::Mask: Copy;
}

impl<const K: usize> SimdStructure<K> for mvtable::Mvt<K, f32> {
    fn collides_simd<const L: usize>(
        &self,
        centers: &[Simd<f32, L>; K],
        radii: Simd<f32, L>,
    ) -> bool
    where
        Simd<f32, L>: AxisSimd<L>,
        <Simd<f32, L> as SimdPartialEq>::Mask: Copy,
    {
        Self::collides_simd(self, centers, radii)
    }
}

impl<const K: usize> SimdStructure<K> for mvtable::MutableMvt<K, f32> {
    fn collides_simd<const L: usize>(
        &self,
        centers: &[Simd<f32, L>; K],
        radii: Simd<f32, L>,
    ) -> bool
    where
        Simd<f32, L>: AxisSimd<L>,
        <Simd<f32, L> as SimdPartialEq>::Mask: Copy,
    {
        Self::collides_simd(self, centers, radii)
    }
}

impl<const K: usize> SimdStructure<K> for capt::Capt<K, f32, u32> {
    fn collides_simd<const L: usize>(
        &self,
        centers: &[Simd<f32, L>; K],
        radii: Simd<f32, L>,
    ) -> bool
    where
        Simd<f32, L>: AxisSimd<L>,
        <Simd<f32, L> as SimdPartialEq>::Mask: Copy,
    {
        Self::collides_simd(self, centers, radii)
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
