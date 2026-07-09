//! Spatial point-cloud downsampling filters.
//!
//! Two filters are implemented, matching the reference preprocessing each of the two structures
//! under comparison actually ships with, so `mbm_bench`'s downsampling matches real usage rather
//! than an ad hoc choice:
//! - [`centervox_filter`], a voxel-grid filter ported from the MVT reference implementation (<https://github.com/chingchennn/vamp_mvt/blob/main/src/impl/vamp/collision/filter_centervox.hh>).
//! - [`morton_filter`], a Morton-order proximity filter ported from `capt`'s reference
//!   implementation (<https://github.com/KavrakiLab/capt/blob/main/morton_filter/src/lib.rs>).

use std::collections::HashMap;

/// Downsample `points` to at most one point per cubic voxel of edge length `voxel_size`, keeping
/// whichever point in each occupied voxel lies closest to that voxel's center.
///
/// This is a straightforward `HashMap`-based port of the MVT reference implementation's
/// `CenterSelectiveVoxelFilter`; the reference's hierarchical X/Y/Z lookup tables and fixed voxel
/// pool exist to keep voxel insertion allocation-free in an online, real-time setting, which
/// doesn't apply to this offline, once-per-benchmark preprocessing step.
#[must_use]
pub fn centervox_filter(points: &[[f32; 3]], voxel_size: f32) -> Vec<[f32; 3]> {
    if points.is_empty() {
        return Vec::new();
    }

    let mut aabb_min = [f32::INFINITY; 3];
    for p in points {
        for k in 0..3 {
            aabb_min[k] = aabb_min[k].min(p[k]);
        }
    }

    let mut voxels: HashMap<(i32, i32, i32), ([f32; 3], f32)> = HashMap::new();
    for &p in points {
        let cell = (
            ((p[0] - aabb_min[0]) / voxel_size).floor() as i32,
            ((p[1] - aabb_min[1]) / voxel_size).floor() as i32,
            ((p[2] - aabb_min[2]) / voxel_size).floor() as i32,
        );
        let center = [
            aabb_min[0] + (cell.0 as f32 + 0.5) * voxel_size,
            aabb_min[1] + (cell.1 as f32 + 0.5) * voxel_size,
            aabb_min[2] + (cell.2 as f32 + 0.5) * voxel_size,
        ];
        let dist_sq: f32 = (0..3).map(|k| (p[k] - center[k]).powi(2)).sum();

        voxels
            .entry(cell)
            .and_modify(|(stored, stored_dist_sq)| {
                if dist_sq < *stored_dist_sq {
                    *stored = p;
                    *stored_dist_sq = dist_sq;
                }
            })
            .or_insert((p, dist_sq));
    }
    voxels.into_values().map(|(p, _)| p).collect()
}

/// Downsample `points` in place by, for every permutation of the 3 axes, sorting by Morton
/// (Z-order) code along that permutation and greedily dropping any point that lands within
/// `min_sep` of the previous (Morton-adjacent) point kept.
///
/// Ported from `capt`'s reference `morton_filter` crate. A single Morton curve has locality gaps
/// (points that are spatially close can land far apart in Morton order at certain boundaries), so
/// the reference implementation repeats the sort-and-filter pass across all 6 axis permutations to
/// catch near-neighbors any single permutation's gaps would miss.
pub fn morton_filter(points: &mut Vec<[f32; 3]>, min_sep: f32) {
    const PERMUTATIONS_3D: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for permutation in PERMUTATIONS_3D {
        filter_permutation(points, min_sep, permutation);
    }
}

/// A single Morton-order filtering pass along one axis `permutation`; see [`morton_filter`].
fn filter_permutation(points: &mut Vec<[f32; 3]>, min_sep: f32, permutation: [usize; 3]) {
    if points.len() < 2 {
        return;
    }

    let mut aabb_min = [f32::INFINITY; 3];
    let mut aabb_max = [f32::NEG_INFINITY; 3];
    for point in points.iter() {
        for k in 0..3 {
            aabb_min[k] = aabb_min[k].min(point[k]);
            aabb_max[k] = aabb_max[k].max(point[k]);
        }
    }
    let rsq = min_sep * min_sep;

    points.sort_by_cached_key(|point| morton_index(point, &aabb_min, &aabb_max, permutation));

    let mut i = 0;
    let mut j = 1;
    while j < points.len() {
        if distsq(&points[i], &points[j]) > rsq {
            i += 1;
            points[i] = points[j];
        }
        j += 1;
    }
    points.truncate(i + 1);
}

fn distsq(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum()
}

/// The Morton (Z-order) code of `point` within `[aabb_min, aabb_max]`, interleaving each axis's
/// bits in the order given by `permutation` (a permutation of `[0, 1, 2]`).
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "matches the reference implementation's own casts, which quantize a normalized \
              [0, 1) coordinate into WIDTH bits"
)]
fn morton_index(
    point: &[f32; 3],
    aabb_min: &[f32; 3],
    aabb_max: &[f32; 3],
    permutation: [usize; 3],
) -> u32 {
    const WIDTH: u32 = u32::BITS / 3;
    const MASK: u32 = 0b001_001_001_001_001_001_001_001_001_001;

    permutation
        .into_iter()
        .enumerate()
        .map(|(i, k)| {
            let extent = aabb_max[k] - aabb_min[k];
            let normalized = if extent > 0.0 {
                (point[k] - aabb_min[k]) / extent
            } else {
                0.0
            };
            pdep((normalized * (1u32 << WIDTH) as f32) as u32, MASK << i)
        })
        .fold(0, |a, b| a | b)
}

/// Deposit the low `mask.count_ones()` bits of `a` into the bit positions set in `mask`, in
/// ascending order (a portable fallback for the `pdep` instruction).
fn pdep(a: u32, mut mask: u32) -> u32 {
    let mut out = 0;
    for i in 0..mask.count_ones() {
        let bit = mask & !(mask - 1);
        if a & (1 << i) != 0 {
            out |= bit;
        }
        mask ^= bit;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{centervox_filter, morton_filter};

    #[test]
    fn morton_one_point() {
        let mut points = vec![[0.0; 3]];
        morton_filter(&mut points, 0.01);
        assert_eq!(points, vec![[0.0; 3]]);
    }

    #[test]
    fn morton_duplicate() {
        let mut points = vec![[0.0; 3]; 2];
        morton_filter(&mut points, 0.01);
        assert_eq!(points, vec![[0.0; 3]]);
    }

    #[test]
    fn morton_too_close() {
        let mut points = vec![[0.0; 3], [0.001; 3]];
        morton_filter(&mut points, 0.01);
        assert_eq!(points, vec![[0.0; 3]]);
    }

    #[test]
    fn morton_too_far() {
        let mut points = vec![[0.0; 3], [0.01; 3]];
        morton_filter(&mut points, 0.01);
        assert_eq!(points, vec![[0.0; 3], [0.01; 3]]);
    }

    #[test]
    fn morton_never_grows_or_loses_all_points() {
        let mut rng_state = 0x1234_5678_u32;
        let mut next = || {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            (rng_state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let original: Vec<[f32; 3]> = (0..500).map(|_| [next(), next(), next()]).collect();

        let mut filtered = original.clone();
        morton_filter(&mut filtered, 0.05);

        assert!(!filtered.is_empty());
        assert!(filtered.len() <= original.len());
    }

    #[test]
    fn centervox_empty() {
        assert_eq!(centervox_filter(&[], 0.1), Vec::<[f32; 3]>::new());
    }

    #[test]
    fn centervox_single_voxel_keeps_one_point() {
        let points = [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]];
        let filtered = centervox_filter(&points, 10.0);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn centervox_keeps_point_closest_to_voxel_center() {
        // A voxel spanning [0, 1) on every axis has its center at (0.5, 0.5, 0.5).
        let points = [[0.5, 0.5, 0.5], [0.01, 0.01, 0.01]];
        let filtered = centervox_filter(&points, 1.0);
        assert_eq!(filtered, vec![[0.5, 0.5, 0.5]]);
    }

    #[test]
    fn centervox_far_apart_points_all_kept() {
        let points = [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [-10.0, -10.0, -10.0]];
        let filtered = centervox_filter(&points, 0.1);
        assert_eq!(filtered.len(), points.len());
    }
}
