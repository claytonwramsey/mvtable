//! Shared utilities for benchmarking and correctness-testing `mvtable` against `capt` and
//! `kiddo`'s immutable k-d tree.
//!
//! The [`Structure`] trait gives all three collision-checking structures a common
//! construct-then-query interface, so that both `benches/collide.rs` and `tests/correctness.rs`
//! can be written once and run against every structure. [`SimdStructure`] additionally covers
//! `mvtable` and `capt`'s SIMD-batched queries.
#![feature(portable_simd)]

use std::{
    num::NonZeroUsize,
    simd::{Simd, cmp::SimdPartialEq},
};

use capt::AxisSimd;
use kiddo::SquaredEuclidean;
use rand::{Rng, RngExt};

pub mod filter;

/// Radius of a robot's largest sphere that actually moves during planning, traced by hand from
/// each robot's spherized URDF.
pub fn mobile_max_radius(robot: &str) -> f32 {
    match robot {
        "panda" => 0.06,
        "ur5" => 0.08,
        "fetch" => 0.15,
        "baxter" => 0.1,
        _ => panic!("no largest-mobile-sphere radius recorded for robot {robot:?}"),
    }
}

/// Radius of each robot's largest sphere on any link.
pub fn robot_max_radius(robot: &str) -> f32 {
    match robot {
        "panda" => 0.08,
        "ur5" => 0.08,
        "fetch" => 0.24,
        "baxter" => 0.5,
        _ => panic!("no largest-mobile-sphere radius recorded for robot {robot:?}"),
    }
}

/// The true maximum collision-query radius `carom`'s generated `fkcc` ever issues for this robot.
pub fn true_max_query_radius(robot: &str) -> f32 {
    match robot {
        "panda" => 0.22,
        "ur5" => 0.35,
        "fetch" => 0.42,
        "baxter" => 0.6,
        _ => panic!("no true max query radius recorded for robot {robot:?}"),
    }
}

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
        // `Mvt::new` takes an explicit voxel width rather than a query-radius range.
        // Using `r_range.1` is mostly a hack.
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
        Self::new_from_slice(points).expect("bucket size 32 is soft-limited, so this can't fail")
    }

    fn collides(&self, center: &[f32; K], radius: f32) -> bool {
        // an empty tree has no nearest neighbor to query for.
        self.size() != 0 && kiddo_collides_one(self, center, radius)
    }
}

/// Test if a kiddo KDT collides with a sphere.
#[inline]
fn kiddo_collides_one<const K: usize>(
    tree: &kiddo::ImmutableKdTree<f32, K>,
    center: &[f32; K],
    radius: f32,
) -> bool {
    !tree
        .query(center)
        .nearest_n::<SquaredEuclidean<f32>>(NonZeroUsize::new(1).unwrap())
        .within(radius * radius)
        .unsorted()
        .execute()
        .is_empty()
}

/// Compute the total memory used (stack + heap) by a `kiddo::ImmutableKdTree<f32, K>`, measured
/// in bytes.
///
/// `kiddo` exposes no such method, and its fields are private, so this is computed analytically
/// from [`kiddo::ImmutableKdTree::size`]. This formula was fit by measuring heap allocations (via a
/// counting `GlobalAlloc`) across every leaf-count boundary up to 100,000 points and K in
/// {2, 3, 4}, and matched exactly at every point tried.
#[must_use]
pub fn kiddo_memory_used<const K: usize>(tree: &kiddo::ImmutableKdTree<f32, K>) -> usize {
    /// `kiddo::ImmutableKdTree<A, K>`'s fixed leaf bucket size (its `B` const-generic param).
    const B: usize = 32;
    /// Bytes per `leaf_extents` entry (`(usize, usize)`, a byte offset and length into the arena).
    const LEAF_EXTENTS_BYTES: usize = 16;
    /// Measured bytes per `Eytzinger` stem entry.
    const STEM_BYTES: usize = 15;

    let item_count = tree.size();
    let leaf_count_raw = item_count.div_ceil(B).max(1);
    // a single leaf needs no stem to route to it; more than one leaf is always padded up to a
    // power of two, to give the Eytzinger stem array simple arithmetic indexing.
    let leaf_count = if leaf_count_raw <= 1 {
        1
    } else {
        leaf_count_raw.next_power_of_two()
    };

    let item_bytes = item_count * (K * size_of::<f32>() + size_of::<u32>());
    let leaf_extents_bytes = leaf_count * LEAF_EXTENTS_BYTES;
    let stem_bytes = if leaf_count >= 2 {
        leaf_count * STEM_BYTES
    } else {
        0
    };
    // small constant overhead measured at the two smallest leaf counts, not explained by the
    // per-item/per-leaf/per-stem terms above.
    let correction = match leaf_count {
        1 => 7,
        2 => 16,
        _ => 0,
    };

    size_of::<kiddo::ImmutableKdTree<f32, K>>()
        + item_bytes
        + leaf_extents_bytes
        + stem_bytes
        + correction
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

/// The vendored C++ reference implementation of the MVT (see `crates/mvt-cpp/vendor/README.md`),
/// hardcoded to 3D like the upstream code it wraps (unlike `mvtable`/`capt`/`kiddo`, which are
/// generic over `K`).
impl Structure<3> for mvt_cpp::MvtCpp {
    const NAME: &'static str = "mvt_cpp";

    fn build(points: &[[f32; 3]], r_range: (f32, f32)) -> Self {
        Self::new(points, r_range)
    }

    fn collides(&self, center: &[f32; 3], radius: f32) -> bool {
        Self::collides(self, center, radius)
    }
}

impl SimdStructure<3> for mvt_cpp::MvtCpp {
    /// Uses the vendored implementation's true vectorized `collides_simd` when `L ==
    /// mvt_cpp::SIMD_WIDTH` but otherwise falls back to another impl.
    fn collides_simd<const L: usize>(
        &self,
        centers: &[Simd<f32, L>; 3],
        radii: Simd<f32, L>,
    ) -> bool
    where
        Simd<f32, L>: AxisSimd<L>,
        <Simd<f32, L> as SimdPartialEq>::Mask: Copy,
    {
        if L != mvt_cpp::SIMD_WIDTH {
            let xs = centers[0].to_array();
            let ys = centers[1].to_array();
            let zs = centers[2].to_array();
            let rs = radii.to_array();
            return (0..L).any(|l| self.collides(&[xs[l], ys[l], zs[l]], rs[l]));
        }

        let to_fixed = |v: [f32; L]| -> [f32; mvt_cpp::SIMD_WIDTH] {
            let mut out = [0.0f32; mvt_cpp::SIMD_WIDTH];
            out.copy_from_slice(&v);
            out
        };
        let centers_fixed = [
            to_fixed(centers[0].to_array()),
            to_fixed(centers[1].to_array()),
            to_fixed(centers[2].to_array()),
        ];
        let radii_fixed = to_fixed(radii.to_array());
        Self::collides_simd(self, &centers_fixed, &radii_fixed)
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
/// boundary of a grid sized with a matching `voxel_width`.
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
