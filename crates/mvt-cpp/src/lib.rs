//! Safe Rust wrapper around the vendored C++ reference implementation of the MVT.
use std::{error::Error, ffi::c_void, fmt};

mod raw {
    use std::ffi::c_void;

    unsafe extern "C" {
        pub fn mvt_cpp_new(
            points_xyz: *const f32,
            n_points: usize,
            min_radius: f32,
            max_radius: f32,
            workspace_min: *const f32,
            workspace_max: *const f32,
            point_radius: f32,
        ) -> *mut c_void;

        pub fn mvt_cpp_free(mvt: *mut c_void);

        pub fn mvt_cpp_collides(mvt: *const c_void, center: *const f32, radius: f32) -> bool;

        pub fn mvt_cpp_collides_simd(
            mvt: *const c_void,
            centers_x: *const f32,
            centers_y: *const f32,
            centers_z: *const f32,
            radii: *const f32,
        ) -> bool;

        pub fn mvt_cpp_simd_width() -> usize;

        pub fn mvt_cpp_memory_used(mvt: *const c_void) -> usize;

        pub fn mvt_cpp_would_overflow(
            points_xyz: *const f32,
            n_points: usize,
            max_radius: f32,
            workspace_min: *const f32,
            workspace_max: *const f32,
        ) -> bool;
    }
}

/// The lane width of [`MvtCpp::collides_simd`]. Must track `build.rs`'s `target_arch` dispatch; a
/// test ([`tests::simd_width_matches_vendored_backend`]) checks it against the vendored code's own
/// `FVectorT::num_scalars` at runtime.
#[cfg(target_arch = "x86_64")]
pub const SIMD_WIDTH: usize = 8;

#[cfg(target_arch = "aarch64")]
pub const SIMD_WIDTH: usize = 4;

/// A point cloud loaded into the vendored C++ MVT (`vamp::collision::MVT`), ready for collision
/// queries.
pub struct MvtCpp {
    ptr: *mut c_void,
}

/// Returned by [`MvtCpp::try_new`] (and, as a panic, from [`MvtCpp::new`]) when `points` would
/// overflow the vendored C++ implementation's hierarchy-table pool for the given `max_radius`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Overflow;

impl fmt::Display for Overflow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "point cloud would overflow the vendored MVT's fixed-capacity pools"
        )
    }
}

impl Error for Overflow {}

// `MVT` owns all of its heap allocations outright and every method callable through this wrapper is
// `const` on the C++ side, so shared read-only access from multiple threads is sound.
unsafe impl Send for MvtCpp {}
unsafe impl Sync for MvtCpp {}

/// A cubic (equal-extent-on-every-axis) workspace bounding box for `points`, padded by
/// `max_radius`.
fn cubic_workspace(points: &[[f32; 3]], max_radius: f32) -> ([f32; 3], [f32; 3]) {
    let Some(first) = points.first() else {
        let half = (2.0 * max_radius).max(1.0);
        return ([-half; 3], [half; 3]);
    };
    let mut lo = *first;
    let mut hi = *first;
    for p in &points[1..] {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let center: [f32; 3] = std::array::from_fn(|k| (lo[k] + hi[k]) / 2.0);
    let extent = (0..3).fold(0.0f32, |m, k| m.max(hi[k] - lo[k]));
    let half = extent.max(4.0 * max_radius) / 2.0 + max_radius;
    (
        std::array::from_fn(|k| center[k] - half),
        std::array::from_fn(|k| center[k] + half),
    )
}

/// Predict whether constructing an `MvtCpp` from `points`/`max_radius` would overflow the
/// vendored C++ implementation's fixed per-voxel point capacity (and so abort the process) or hit
/// its `grid_width == 0` table-corruption case, without actually allocating any of its pools.
#[must_use]
fn would_overflow(points: &[[f32; 3]], max_radius: f32) -> bool {
    let (workspace_min, workspace_max) = cubic_workspace(points, max_radius);
    let points_xyz = points.as_ptr().cast::<f32>();
    unsafe {
        raw::mvt_cpp_would_overflow(
            points_xyz,
            points.len(),
            max_radius,
            workspace_min.as_ptr(),
            workspace_max.as_ptr(),
        )
    }
}

impl MvtCpp {
    /// Build a new `MvtCpp` containing `points`, for queries with radius in `r_range` (`(r_min,
    /// r_max)`).
    ///
    /// # Panics
    ///
    /// Panics if `points` would overflow the vendored implementation's fixed-capacity pools for
    /// `r_range.1`. Use [`try_new`](MvtCpp::try_new) to handle that case instead of panicking.
    #[must_use]
    pub fn new(points: &[[f32; 3]], r_range: (f32, f32)) -> Self {
        Self::try_new(points, r_range).expect("failed to construct MvtCpp; see mvt_cpp::Overflow")
    }

    /// Build a new `MvtCpp` containing `points`, for queries with radius in `r_range` (`(r_min,
    /// r_max)`), checking first whether doing so would overflow the vendored implementation's
    /// fixed-capacity pools (see [`Overflow`]) instead of risking the process abort that would
    /// otherwise follow.
    pub fn try_new(points: &[[f32; 3]], r_range: (f32, f32)) -> Result<Self, Overflow> {
        let (min_radius, max_radius) = r_range;
        debug_assert!(max_radius > 0.0, "MvtCpp requires a positive max radius");
        debug_assert_eq!(
            unsafe { raw::mvt_cpp_simd_width() },
            SIMD_WIDTH,
            "vendored MVT's FVectorT lane width no longer matches SIMD_WIDTH"
        );
        if would_overflow(points, max_radius) {
            return Err(Overflow);
        }
        let (workspace_min, workspace_max) = cubic_workspace(points, max_radius);
        let points_xyz = points.as_ptr().cast::<f32>();
        let ptr = unsafe {
            raw::mvt_cpp_new(
                points_xyz,
                points.len(),
                min_radius,
                max_radius,
                workspace_min.as_ptr(),
                workspace_max.as_ptr(),
                0.0,
            )
        };
        assert!(!ptr.is_null(), "mvt_cpp_new returned null");
        Ok(Self { ptr })
    }

    #[must_use]
    pub fn collides(&self, center: &[f32; 3], radius: f32) -> bool {
        unsafe { raw::mvt_cpp_collides(self.ptr, center.as_ptr(), radius) }
    }

    /// Determine whether any point lies within the corresponding lane of `radii` of the
    /// corresponding lane of `centers`, [`SIMD_WIDTH`]-wide.
    #[must_use]
    pub fn collides_simd(
        &self,
        centers: &[[f32; SIMD_WIDTH]; 3],
        radii: &[f32; SIMD_WIDTH],
    ) -> bool {
        unsafe {
            raw::mvt_cpp_collides_simd(
                self.ptr,
                centers[0].as_ptr(),
                centers[1].as_ptr(),
                centers[2].as_ptr(),
                radii.as_ptr(),
            )
        }
    }

    #[must_use]
    pub fn memory_used(&self) -> usize {
        unsafe { raw::mvt_cpp_memory_used(self.ptr) }
    }
}

impl Drop for MvtCpp {
    fn drop(&mut self) {
        unsafe { raw::mvt_cpp_free(self.ptr) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_force(points: &[[f32; 3]], center: &[f32; 3], radius: f32) -> bool {
        let rsq = radius * radius;
        points.iter().any(|p| {
            let d: [f32; 3] = std::array::from_fn(|k| p[k] - center[k]);
            d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= rsq
        })
    }

    #[test]
    fn matches_brute_force_within_r_max() {
        // A simple linear congruential generator, to avoid pulling in `rand` for one test.
        let mut state = 88172645463325252u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        };
        let coord = |next: &mut dyn FnMut() -> f32| (next() - 0.5) * 10.0;

        let points: Vec<[f32; 3]> = (0..800)
            .map(|_| [coord(&mut next), coord(&mut next), coord(&mut next)])
            .collect();
        let r_max = 0.2;
        let mvt = MvtCpp::try_new(&points, (0.0, r_max)).expect("test data shouldn't overflow");

        for _ in 0..5000 {
            let center = [coord(&mut next), coord(&mut next), coord(&mut next)];
            let radius = next() * r_max;
            assert_eq!(
                mvt.collides(&center, radius),
                brute_force(&points, &center, radius),
                "disagreed with brute force: center={center:?}, radius={radius}"
            );
        }
    }

    #[test]
    fn simd_width_matches_vendored_backend() {
        assert_eq!(unsafe { raw::mvt_cpp_simd_width() }, SIMD_WIDTH);
    }

    /// a query with radius beyond `r_max`, or a center far outside the workspace, used to corrupt
    /// `MVT`'s internal table lookups.
    #[test]
    fn out_of_range_queries_return_false_instead_of_crashing() {
        let points = vec![[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0]];
        let r_max = 0.08;
        let mvt = MvtCpp::new(&points, (0.0, r_max));

        // Radius beyond r_max, center still near the cloud.
        assert!(!mvt.collides(&[0.0, 0.0, 0.0], 5.0));
        // In-range radius, but a center wildly outside the workspace.
        assert!(!mvt.collides(&[1e6, -1e6, 1e6], 0.05));
        // Both at once.
        assert!(!mvt.collides(&[-1e8, 1e8, -1e8], 50.0));

        let centers = [[0.0; SIMD_WIDTH]; 3];
        let mut radii = [0.01f32; SIMD_WIDTH];
        radii[SIMD_WIDTH - 1] = 5.0; // one out-of-range lane amid otherwise-safe ones
        assert!(mvt.collides_simd(&centers, &radii));

        // All lanes far from every point (so none should collide), one lane wildly out of range.
        let mut centers_bad = [[100.0; SIMD_WIDTH]; 3];
        centers_bad[0][SIMD_WIDTH - 1] = 1e6;
        let radii_safe = [0.01f32; SIMD_WIDTH];
        assert!(!mvt.collides_simd(&centers_bad, &radii_safe));
    }

    #[test]
    fn empty_cloud_never_collides() {
        let mvt = MvtCpp::new(&[], (0.0, 0.1));
        assert!(!mvt.collides(&[0.0, 0.0, 0.0], 0.0));
        assert!(!mvt.collides(&[1.0, 2.0, 3.0], 100.0));
    }

    /// The vendored implementation's remaining overflow mode.
    #[test]
    fn spread_out_cloud_overflow_returns_err_instead_of_aborting() {
        let max_radius = 0.1;
        // A workspace this wide (from `cubic_workspace`'s minimal padding) gives grid_width =
        // floor(5.2 / 0.1) = 52, so `grid_width^2 * 0.8` = 2163 columns of headroom.
        let n = 53;
        let step = 5.0 / (n - 1) as f32;
        let points: Vec<[f32; 3]> = (0..n)
            .flat_map(|i| (0..n).map(move |j| [i as f32 * step, j as f32 * step, 0.0]))
            .collect();
        // 53 * 53 = 2809 distinct (x, y) columns - comfortably past that 2163 estimate.
        assert!(MvtCpp::try_new(&points, (0.0, max_radius)).is_err());
    }
}
