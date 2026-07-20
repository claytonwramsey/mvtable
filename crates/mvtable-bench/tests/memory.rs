//! Compares `Mvt::memory_used()` against `MutableMvt::memory_used()` for identical point clouds.

use mvtable_bench::{clustered_cloud, uniform_cloud};
use rand::{SeedableRng, rngs::SmallRng};

fn compare_memory(points: &[[f32; 3]], r_max: f32) -> (usize, usize) {
    let mvt = mvtable::Mvt::<3, f32>::new(points, r_max);
    let mvt_mutable = mvtable::MutableMvt::<3, f32>::new(points, r_max);
    (mvt.memory_used(), mvt_mutable.memory_used())
}

#[test]
fn mutable_uses_more_memory_uniform() {
    let mut rng = SmallRng::seed_from_u64(1);
    let points = uniform_cloud::<_, 3>(&mut rng, 20_000, 5.0);
    let (immutable_bytes, mutable_bytes) = compare_memory(&points, 0.05);

    println!(
        "uniform: immutable={immutable_bytes} bytes, mutable={mutable_bytes} bytes, \
         ratio={:.2}x",
        mutable_bytes as f64 / immutable_bytes as f64
    );
    assert!(
        mutable_bytes > immutable_bytes,
        "MutableMvt ({mutable_bytes} bytes) should use more memory than Mvt \
         ({immutable_bytes} bytes) for the same uniform point cloud"
    );
}

#[test]
fn mutable_uses_more_memory_clustered() {
    // small, tight clusters: most voxels hold very few points, which is the worst case for
    // `MutableMvt`'s per-voxel `Vec` capacity slack and per-voxel allocation overhead.
    let mut rng = SmallRng::seed_from_u64(2);
    let points = clustered_cloud::<_, 3>(&mut rng, 20_000, 400, 5.0, 0.02);
    let (immutable_bytes, mutable_bytes) = compare_memory(&points, 0.05);

    println!(
        "clustered: immutable={immutable_bytes} bytes, mutable={mutable_bytes} bytes, \
         ratio={:.2}x",
        mutable_bytes as f64 / immutable_bytes as f64
    );
    assert!(
        mutable_bytes > immutable_bytes,
        "MutableMvt ({mutable_bytes} bytes) should use more memory than Mvt \
         ({immutable_bytes} bytes) for the same clustered point cloud"
    );
}

/// `kiddo_memory_used`'s stem/leaf-count formulas branch on bucket-boundary crossings (`B` = 32,
/// see its doc comment), so exercise those boundaries plus the empty tree rather than just one
/// arbitrary size.
#[test]
fn kiddo_memory_used_grows_monotonically_across_bucket_boundaries() {
    let mut rng = SmallRng::seed_from_u64(3);
    let points = uniform_cloud::<_, 3>(&mut rng, 20_000, 5.0);

    let mut prev_bytes = 0;
    for n in [1, 2, 31, 32, 33, 63, 64, 65, 1000, 20_000] {
        let tree = kiddo::ImmutableKdTree::<f32, 3>::new_from_slice(&points[..n]);
        let bytes = mvtable_bench::kiddo_memory_used(&tree);
        assert!(
            bytes > prev_bytes,
            "kiddo_memory_used should strictly increase with n, got {bytes} bytes at n={n} \
             (previous n's value was {prev_bytes})"
        );
        prev_bytes = bytes;
    }
}

/// The empty tree is a degenerate edge case (see the doc comment above) - just check it doesn't
/// panic and produces a positive byte count, rather than folding it into the monotonicity sweep.
#[test]
fn kiddo_memory_used_empty_tree_does_not_panic() {
    let tree = kiddo::ImmutableKdTree::<f32, 3>::new_from_slice(&[]);
    assert!(mvtable_bench::kiddo_memory_used(&tree) > 0);
}
