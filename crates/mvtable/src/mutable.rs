//! [`MutableMvt`], a variant of [`Mvt`](crate::Mvt) that supports inserting new points after
//! construction, at the cost of performance.

use alloc::vec::Vec;
use core::{array, fmt, mem::size_of};

#[cfg(feature = "simd")]
use core::simd::{Simd, cmp::SimdPartialOrd};

use crate::{Aabb, Axis, Index, grid};
#[cfg(feature = "simd")]
use crate::{AxisSimd, AxisSimdElement};

#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
/// The errors that can occur when calling [`MutableMvt::try_new`],
/// [`MutableMvt::try_with_point_radius`], or [`MutableMvt::try_with_workspace`].
///
/// # Examples
///
/// ```
/// let points = [[0.0]];
/// let err = mvtable::MutableMvt::<1>::try_new(&points, -1.0).unwrap_err();
/// assert_eq!(err, mvtable::NewMutableMvtError::InvalidVoxelWidth);
/// ```
pub enum NewMutableMvtError {
    /// At least one of the points had a non-finite value.
    NonFinite,
    /// `voxel_width` was not a positive, finite value, so voxels could not be sized.
    InvalidVoxelWidth,
    /// There were too many voxels or points to be represented without integer overflow.
    TooManyVoxels,
    /// [`MutableMvt::try_with_workspace`] was called with `lo[k] > hi[k]` for some axis `k`, so
    /// no valid workspace box exists.
    InvalidWorkspace,
    /// [`MutableMvt::try_new`] or [`MutableMvt::try_with_point_radius`] was called with an empty
    /// point cloud, which has no bounding box to size the grid from. Use
    /// [`MutableMvt::try_with_workspace`] instead to construct an empty `MutableMvt` with
    /// explicit workspace bounds.
    EmptyPointCloud,
}

impl From<grid::TooManyVoxels> for NewMutableMvtError {
    fn from(_: grid::TooManyVoxels) -> Self {
        Self::TooManyVoxels
    }
}

impl fmt::Display for NewMutableMvtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => write!(f, "at least one point had a non-finite value"),
            Self::InvalidVoxelWidth => {
                write!(f, "voxel_width was not a positive, finite value")
            }
            Self::TooManyVoxels => {
                write!(
                    f,
                    "too many voxels or points for the index type to represent"
                )
            }
            Self::InvalidWorkspace => {
                write!(
                    f,
                    "lo[k] > hi[k] for some axis k, so no valid workspace box exists"
                )
            }
            Self::EmptyPointCloud => write!(
                f,
                "the point cloud was empty, so no workspace bounds could be inferred from it"
            ),
        }
    }
}

impl core::error::Error for NewMutableMvtError {}

#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
/// The errors that can occur when calling [`MutableMvt::insert`] or
/// [`MutableMvt::insert_points`].
///
/// # Examples
///
/// ```
/// let mut mvt = mvtable::MutableMvt::<2>::new(&[[0.0, 0.0]], 1.0);
/// let err = mvt.insert(&[f32::NAN, 0.0]).unwrap_err();
/// assert_eq!(err, mvtable::InsertError::NonFinite);
/// ```
pub enum InsertError {
    /// At least one of the points had a non-finite value.
    NonFinite,
    /// There were too many voxels or points to be represented without integer overflow.
    TooManyVoxels,
}

impl From<grid::TooManyVoxels> for InsertError {
    fn from(_: grid::TooManyVoxels) -> Self {
        Self::TooManyVoxels
    }
}

impl fmt::Display for InsertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => write!(f, "the point had a non-finite value"),
            Self::TooManyVoxels => {
                write!(
                    f,
                    "too many voxels or points for the index type to represent"
                )
            }
        }
    }
}

impl core::error::Error for InsertError {}

/// Per-voxel storage for [`MutableMvt`].
#[derive(Clone, Debug)]
struct MutableVoxel<A, const K: usize> {
    /// A local bounding box over the points contained by this voxel.
    aabb: Aabb<A, K>,
    /// This voxel's points, in struct-of-arrays order: `axes[k]` holds the `k`-th coordinate of
    /// every point in the voxel, in insertion order. Every `axes[k]` always has the same length.
    axes: [Vec<A>; K],
}

impl<A: Axis, const K: usize> MutableVoxel<A, K> {
    /// A voxel containing no points.
    fn empty() -> Self {
        Self {
            aabb: Aabb::EMPTY,
            axes: array::from_fn(|_| Vec::new()),
        }
    }

    /// The number of points currently stored in this voxel.
    const fn count(&self) -> usize {
        self.axes[0].len()
    }
}

#[derive(Clone, Debug)]
/// A variant of [`Mvt`](crate::Mvt) that supports inserting new points after construction.
///
/// # Performance
///
/// [`Mvt`](crate::Mvt) gets its memory density from
/// flattening every voxel's points into one immutable, exactly-sized, contiguous
/// struct-of-arrays pool at construction. That layout can't support inserting a point into an
/// arbitrary voxel without shifting every later voxel's data, so `MutableMvt` instead gives every
/// voxel its own independently growable per-axis `Vec<A>`s.
///
/// The mutability of `MutableMvt` comes with performance costs to memory and construction time.
///
/// - `MutableMvt` has a significant memory overhead to `Mvt`, to a factor of about 2x. The memory
///   overhead is greatest for sparse point clouds with few points.
/// - Construction is typically 1.3x-1.8x slower.
/// - Query throughput is generally about even between `Mvt` and `MutableMvt`. Users sensitive to
///   performance should measure their own workloads.
///
/// # Workspace bounds
///
/// Like [`Mvt`](crate::Mvt), a `MutableMvt`'s grid is sized once, at construction, and that
/// sizing is then fixed for the rest of the structure's life. [`Self::new`]/
/// [`Self::with_point_radius`] size the grid from the given point cloud, so they require it to
/// be non-empty. [`Self::with_workspace`] instead lets you set the bounds explicitly, so it can
/// construct an empty `MutableMvt`. Either way, a `MutableMvt` always has established workspace
/// bounds, so [`Self::insert`]/[`Self::insert_points`] can never fail for lack of them. Points
/// inserted later that fall outside the original workspace are not rejected, but the process for
/// handling them activates slower paths that reduce query performance. For best performance,
/// construct a `MutableMvt` with a representative initial point cloud (or workspace box) spanning
/// the region future insertions will land in.
///
/// # Examples
///
/// ```rust
/// use mvtable::MutableMvt;
///
/// let points = [[0.0, 1.1], [0.2, 3.1]];
/// let mut mvt = MutableMvt::<2>::new(&points, 2.0);
/// assert!(!mvt.collides(&[10.0, 10.0], 1.0));
///
/// mvt.insert(&[10.0, 10.0])?;
/// assert!(mvt.collides(&[10.0, 10.0], 1.0));
/// # Ok::<(), mvtable::InsertError>(())
/// ```
pub struct MutableMvt<const K: usize, A = f32, I = u32> {
    /// The number of voxels along each axis of the grid, fixed at construction.
    grid_width: [I; K],
    /// The number of grid cells per unit length along each axis.
    scale: [A; K],
    /// The frozen coordinate-transform origin used to map points into grid coordinates, fixed at
    /// construction. Distinct from `global_aabb`, which keeps growing as points are inserted:
    /// `grid_lo` must stay fixed for previously-inserted points' grid coordinates to remain valid.
    grid_lo: [A; K],
    /// The radius to add to every point to account for its physical volume.
    r_point: A,
    /// A bounding box over every point inserted so far, used to quickly reject far-away queries.
    /// Unlike `grid_lo`/`scale`, this grows to include every inserted point.
    global_aabb: Aabb<A, K>,
    /// The sparse table hierarchy, in the same format as [`Mvt`](crate::Mvt)'s, but growable.
    tables: Vec<I>,
    /// Metadata and owned point storage for each occupied voxel.
    voxels: Vec<MutableVoxel<A, K>>,
}

impl<const K: usize, A: Axis, I: Index> MutableMvt<K, A, I> {
    /// Construct a new `MutableMvt` containing all the points in `points`.
    ///
    /// `voxel_width` sizes the grid's voxels.
    /// Good values for `voxel_width` are best found by benchmarking your own workload.
    ///
    /// # Panics
    ///
    /// This function will panic if `points` is empty, if any point contains a non-finite value,
    /// or if `voxel_width` is not a positive, finite value. To construct an empty `MutableMvt`,
    /// use [`Self::with_workspace`] instead.
    ///
    /// # Examples
    ///
    /// ```
    /// let points = [[0.0]];
    /// let mvt = mvtable::MutableMvt::<1>::new(&points, f32::INFINITY);
    /// assert!(mvt.collides(&[1.0], 1.5));
    /// ```
    #[must_use]
    pub fn new(points: &[[A; K]], voxel_width: A) -> Self {
        Self::try_new(points, voxel_width)
            .expect("failed to construct MutableMvt; see NewMutableMvtError variants")
    }

    /// Construct a new `MutableMvt` containing all the points in `points`, with a point radius
    /// `r_point` added to every query.
    ///
    /// # Panics
    ///
    /// This function will panic if `points` is empty, if any point contains a non-finite value,
    /// or if `voxel_width` is not a positive, finite value. To construct an empty `MutableMvt`,
    /// use [`Self::with_workspace`] instead.
    ///
    /// # Examples
    ///
    /// ```
    /// let points = [[0.0]];
    /// let mvt = mvtable::MutableMvt::<1>::with_point_radius(&points, f32::INFINITY, 0.2);
    /// assert!(mvt.collides(&[1.0], 1.5));
    /// ```
    #[must_use]
    pub fn with_point_radius(points: &[[A; K]], voxel_width: A, r_point: A) -> Self {
        Self::try_with_point_radius(points, voxel_width, r_point)
            .expect("failed to construct MutableMvt; see NewMutableMvtError variants")
    }

    /// Construct a new `MutableMvt` containing all the points in `points`, checking for invalid
    /// input.
    ///
    /// # Errors
    ///
    /// See [`NewMutableMvtError`] for the circumstances in which this function returns an error.
    ///
    /// # Examples
    ///
    /// ```
    /// let points = [[0.0]];
    /// let mvt = mvtable::MutableMvt::<1>::try_new(&points, f32::INFINITY)?;
    /// # Ok::<(), mvtable::NewMutableMvtError>(())
    /// ```
    pub fn try_new(points: &[[A; K]], voxel_width: A) -> Result<Self, NewMutableMvtError> {
        Self::try_with_point_radius(points, voxel_width, A::ZERO)
    }

    /// Construct a new `MutableMvt` containing all the points in `points`, with a point radius
    /// `r_point` added to every query, checking for invalid input.
    ///
    /// # Errors
    ///
    /// See [`NewMutableMvtError`] for the circumstances in which this function returns an error.
    ///
    /// # Examples
    ///
    /// ```
    /// let points = [[0.0]];
    /// let mvt = mvtable::MutableMvt::<1>::try_with_point_radius(&points, f32::INFINITY, 0.01)?;
    /// # Ok::<(), mvtable::NewMutableMvtError>(())
    /// ```
    pub fn try_with_point_radius(
        points: &[[A; K]],
        voxel_width: A,
        r_point: A,
    ) -> Result<Self, NewMutableMvtError> {
        const { assert!(K > 0, "MutableMvt requires at least one dimension") };

        if points.iter().any(|p| p.iter().any(|x| !x.is_finite())) {
            return Err(NewMutableMvtError::NonFinite);
        }
        if voxel_width <= A::ZERO {
            return Err(NewMutableMvtError::InvalidVoxelWidth);
        }

        // an empty point cloud has no bounding box to size the grid from; construct with
        // `Self::with_workspace` instead if an empty `MutableMvt` is wanted.
        let Some(bounding_box) = Aabb::bounding_box(points) else {
            return Err(NewMutableMvtError::EmptyPointCloud);
        };

        let mut this =
            Self::try_with_workspace(bounding_box.lo, bounding_box.hi, voxel_width, r_point)?;
        // every point was already checked finite above, and the grid is already established, so
        // `insert_initialized` is sufficient and cannot fail except via `TooManyVoxels`.
        for p in points {
            this.insert_initialized(p)?;
        }
        Ok(this)
    }

    /// Construct a new, empty `MutableMvt` whose workspace bounds are set directly to the
    /// axis-aligned box from `lo` to `hi`, rather than being inferred from a point cloud, with a
    /// point radius `r_point` added to every query.
    ///
    /// `voxel_width` sizes the grid's voxels, exactly as in [`Self::with_point_radius`].
    ///
    /// # Panics
    ///
    /// This function will panic if any component of `lo` or `hi` is non-finite, if `lo[k] >
    /// hi[k]` for any axis `k`, or if `voxel_width` is not a positive, finite value.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut mvt = mvtable::MutableMvt::<2>::with_workspace([0.0, 0.0], [10.0, 10.0], 1.0, 0.0);
    /// mvt.insert(&[5.0, 5.0])?;
    /// assert!(mvt.collides(&[5.0, 5.0], 0.1));
    /// # Ok::<(), mvtable::InsertError>(())
    /// ```
    #[must_use]
    pub fn with_workspace(lo: [A; K], hi: [A; K], voxel_width: A, r_point: A) -> Self {
        Self::try_with_workspace(lo, hi, voxel_width, r_point)
            .expect("failed to construct MutableMvt; see NewMutableMvtError variants")
    }

    /// Construct a new, empty `MutableMvt` whose workspace bounds are set directly to the
    /// axis-aligned box from `lo` to `hi`, checking for invalid input.
    ///
    /// # Errors
    ///
    /// See [`NewMutableMvtError`] for the circumstances in which this function returns an error.
    ///
    /// # Examples
    ///
    /// ```
    /// let mvt = mvtable::MutableMvt::<2>::try_with_workspace([0.0, 0.0], [10.0, 10.0], 1.0, 0.0)?;
    /// # Ok::<(), mvtable::NewMutableMvtError>(())
    /// ```
    pub fn try_with_workspace(
        lo: [A; K],
        hi: [A; K],
        voxel_width: A,
        r_point: A,
    ) -> Result<Self, NewMutableMvtError> {
        const { assert!(K > 0, "MutableMvt requires at least one dimension") };

        if lo.iter().chain(&hi).any(|x| !x.is_finite()) {
            return Err(NewMutableMvtError::NonFinite);
        }
        if (0..K).any(|k| lo[k] > hi[k]) {
            return Err(NewMutableMvtError::InvalidWorkspace);
        }
        if voxel_width <= A::ZERO {
            return Err(NewMutableMvtError::InvalidVoxelWidth);
        }

        let bounding_box = Aabb { lo, hi };
        let (grid_width, grid_width_i, scale) = grid::size_grid(&bounding_box, voxel_width)?;
        Ok(Self {
            grid_width: grid_width_i,
            scale,
            grid_lo: lo,
            r_point,
            global_aabb: Aabb::EMPTY,
            tables: grid::new_root_table(grid_width),
            voxels: Vec::new(),
        })
    }

    /// Insert `point` into this `MutableMvt`'s grid; `point` must already be known finite.
    fn insert_initialized(&mut self, point: &[A; K]) -> Result<(), grid::TooManyVoxels> {
        let grid_width: [usize; K] = array::from_fn(|k| self.grid_width[k].to_usize());
        let coords = grid::point_to_grid_coords(point, self.grid_lo, self.scale, grid_width);
        let leaf_slot = grid::get_leaf(&mut self.tables, grid_width, coords)?;

        let voxel_idx = if self.tables[leaf_slot] == I::SENTINEL {
            let idx = self.voxels.len();
            self.voxels.push(MutableVoxel::empty());
            self.tables[leaf_slot] = I::from_usize(idx).ok_or(grid::TooManyVoxels)?;
            idx
        } else {
            self.tables[leaf_slot].to_usize()
        };

        let voxel = &mut self.voxels[voxel_idx];
        voxel.aabb.insert(point);
        for (axis, &x) in voxel.axes.iter_mut().zip(point) {
            axis.push(x);
        }
        self.global_aabb.insert(point);

        Ok(())
    }

    /// Insert a single new point into this `MutableMvt`.
    ///
    /// # Errors
    ///
    /// Returns [`InsertError::NonFinite`] if `point` contains a non-finite value, or
    /// [`InsertError::TooManyVoxels`] if inserting `point` would need more voxels or points than
    /// the index type `I` can represent.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut mvt = mvtable::MutableMvt::<2>::new(&[[0.0, 0.0]], 1.0);
    /// mvt.insert(&[5.0, 5.0])?;
    /// assert!(mvt.collides(&[5.0, 5.0], 0.1));
    /// # Ok::<(), mvtable::InsertError>(())
    /// ```
    pub fn insert(&mut self, point: &[A; K]) -> Result<(), InsertError> {
        if point.iter().any(|x| !x.is_finite()) {
            return Err(InsertError::NonFinite);
        }
        Ok(self.insert_initialized(point)?)
    }

    /// Insert every point in `points` into this `MutableMvt`.
    ///
    /// Equivalent to calling [`Self::insert`] once per point, in order. If an error occurs partway
    /// through, every point before the failing one has already been inserted (this method is not
    /// transactional).
    ///
    /// # Errors
    ///
    /// See [`Self::insert`].
    ///
    /// # Examples
    ///
    /// ```
    /// let mut mvt = mvtable::MutableMvt::<2>::new(&[[0.0, 0.0]], 1.0);
    /// mvt.insert_points(&[[5.0, 5.0], [-5.0, -5.0]])?;
    /// assert!(mvt.collides(&[5.0, 5.0], 0.1));
    /// assert!(mvt.collides(&[-5.0, -5.0], 0.1));
    /// # Ok::<(), mvtable::InsertError>(())
    /// ```
    pub fn insert_points(&mut self, points: &[[A; K]]) -> Result<(), InsertError> {
        for p in points {
            self.insert(p)?;
        }
        Ok(())
    }

    /// Look up the voxel containing grid coordinates `coords`, if it is occupied.
    fn lookup_voxel(&self, coords: &[usize; K]) -> Option<&MutableVoxel<A, K>> {
        let mut table_offset = 0usize;
        for &coord in &coords[..K - 1] {
            let next = self.tables[table_offset + coord];
            if next == I::SENTINEL {
                return None;
            }
            table_offset = next.to_usize();
        }
        let leaf = self.tables[table_offset + coords[K - 1]];
        (leaf != I::SENTINEL).then(|| &self.voxels[leaf.to_usize()])
    }

    #[must_use]
    /// Determine whether any point in this tree is within a distance of `radius` to `center`.
    ///
    /// # Examples
    ///
    /// ```
    /// let points = [[0.0; 3], [1.0, -1.1, 0.5], [-0.2, -0.3, 0.25]];
    /// let mvt = mvtable::MutableMvt::<3>::new(&points, 0.2);
    ///
    /// assert!(mvt.collides(&[0.0, 0.1, 0.0], 0.11));
    /// assert!(!mvt.collides(&[0.0, 0.1, 0.0], 0.05));
    /// ```
    pub fn collides(&self, center: &[A; K], radius: A) -> bool {
        if self.voxels.is_empty() {
            return false;
        }
        let r = radius + self.r_point;
        let rsq = r.square();
        if self.global_aabb.closest_distsq_to(center) > rsq {
            return false;
        }
        self.search_block(center, r, |voxel| {
            let count = voxel.count();
            let axes: [&[A]; K] = array::from_fn(|k| &voxel.axes[k][..count]);
            crate::scan_block::<A, K, { crate::SCAN_BLOCK }>(&axes, center, rsq)
        })
    }

    /// Search the block of voxels that could contain a point within `r` (already including
    /// `r_point`) of `center`, calling `check_voxel` on each voxel whose local AABB could contain
    /// such a point. Returns `true` as soon as `check_voxel` does, and `false` if every voxel in
    /// the block has been checked without one returning `true`.
    fn search_block(
        &self,
        center: &[A; K],
        r: A,
        check_voxel: impl Fn(&MutableVoxel<A, K>) -> bool,
    ) -> bool {
        let rsq = r.square();

        let mut bmin = [0usize; K];
        let mut bmax = [0usize; K];
        for k in 0..K {
            let grid_max = self.grid_width[k].to_usize() - 1;
            // theoretically has epsilon-scale errors, but is ok
            let rg = r * self.scale[k];
            let v = (center[k] - self.grid_lo[k]) * self.scale[k];
            bmin[k] = (v - rg).to_index().min(grid_max);
            bmax[k] = (v + rg).to_index().min(grid_max);
        }

        let mut coords = bmin;
        loop {
            if let Some(voxel) = self.lookup_voxel(&coords)
                && voxel.aabb.closest_distsq_to(center) <= rsq
                && check_voxel(voxel)
            {
                return true;
            }

            // odometer-style increment over the K-dimensional search block.
            let mut dim = 0;
            loop {
                if dim == K {
                    return false;
                }
                coords[dim] += 1;
                if coords[dim] <= bmax[dim] {
                    break;
                }
                coords[dim] = bmin[dim];
                dim += 1;
            }
        }
    }

    /// Get an iterator over the points stored in this `MutableMvt`.
    /// It makes no guarantee of iteration order.
    ///
    /// ```
    /// let mvt = mvtable::MutableMvt::<2>::new(&[[0.0, 1.0]], f32::INFINITY);
    /// for point in mvt.points() {
    ///     println!("{point:?}");
    /// }
    /// ```
    pub fn points(&self) -> impl Iterator<Item = [A; K]> + '_ {
        self.voxels.iter().flat_map(move |v| {
            let count = v.count();
            (0..count).map(move |i| array::from_fn(|k| v.axes[k][i]))
        })
    }

    #[must_use]
    #[doc(hidden)]
    /// Get the total memory used (stack + heap) by this structure, measured in bytes.
    /// This function should not be considered stable; it is only used internally for benchmarks.
    pub fn memory_used(&self) -> usize {
        size_of::<Self>()
            + self.tables.capacity() * size_of::<I>()
            + self.voxels.capacity() * size_of::<MutableVoxel<A, K>>()
            + self
                .voxels
                .iter()
                .map(|v| v.axes.iter().map(Vec::capacity).sum::<usize>() * size_of::<A>())
                .sum::<usize>()
    }
}

#[cfg(feature = "simd")]
impl<const K: usize, A: AxisSimdElement, I: Index> MutableMvt<K, A, I> {
    #[must_use]
    /// Determine whether any sphere in a SIMD batch of `L` spheres intersects a point in this
    /// table.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(portable_simd)]
    /// use std::simd::Simd;
    ///
    /// let points = [[1.0, 2.0], [1.1, 1.1]];
    /// let mvt = mvtable::MutableMvt::<2>::new(&points, 0.1);
    ///
    /// let centers = [
    ///     Simd::from_array([1.0, 1.1, 1.2, 1.3]), // x-positions
    ///     Simd::from_array([1.0, 1.1, 1.2, 1.3]), // y-positions
    /// ];
    /// let radii = Simd::splat(0.05);
    ///
    /// assert!(mvt.collides_simd(&centers, radii));
    /// ```
    pub fn collides_simd<const L: usize>(
        &self,
        centers: &[Simd<A, L>; K],
        radii: Simd<A, L>,
    ) -> bool
    where
        Simd<A, L>: AxisSimd<L>,
    {
        if self.voxels.is_empty() {
            return false;
        }

        let r = radii + Simd::splat(self.r_point);
        let rsq = r * r;

        // vectorized global AABB cull across the whole batch at once
        let mut distsq = Simd::splat(A::ZERO);
        for (k, &center) in centers.iter().enumerate() {
            let lo = Simd::splat(self.global_aabb.lo[k]);
            let hi = Simd::splat(self.global_aabb.hi[k]);
            let below = center.simd_lt(lo);
            let above = center.simd_gt(hi);
            let clamped = Simd::<A, L>::select(below, lo, Simd::<A, L>::select(above, hi, center));
            let diff = center - clamped;
            distsq += diff * diff;
        }
        let inbounds = Simd::<A, L>::mask_to_array(distsq.simd_le(rsq));
        if !inbounds.iter().any(|&b| b) {
            return false;
        }

        let r_arr = r.to_array();
        let centers_arr: [[A; L]; K] = array::from_fn(|k| centers[k].to_array());
        (0..L).any(|lane| {
            // this lane was already ruled out by the batched global AABB cull above.
            if !inbounds[lane] {
                return false;
            }
            let center: [A; K] = array::from_fn(|k| centers_arr[k][lane]);
            let r_lane = r_arr[lane];
            let rsq_lane = r_lane.square();
            self.search_block(&center, r_lane, |voxel| {
                Self::points_collide_simd::<L>(voxel, &center, rsq_lane)
            })
        })
    }

    /// Determine whether any of the points stored in `voxel` are within a squared distance of
    /// `rsq` from `center`, checking `L` points at a time.
    fn points_collide_simd<const L: usize>(
        voxel: &MutableVoxel<A, K>,
        center: &[A; K],
        rsq: A,
    ) -> bool
    where
        Simd<A, L>: AxisSimd<L>,
    {
        let count = voxel.count();
        let center_simd: [Simd<A, L>; K] = array::from_fn(|k| Simd::splat(center[k]));
        let rsq_simd = Simd::splat(rsq);

        let mut i = 0;
        while i + L <= count {
            let mut distsq = Simd::splat(A::ZERO);
            for (k, &c) in center_simd.iter().enumerate() {
                let chunk = Simd::from_slice(&voxel.axes[k][i..]);
                let diff = chunk - c;
                distsq += diff * diff;
            }
            if Simd::<A, L>::mask_any(distsq.simd_le(rsq_simd)) {
                return true;
            }
            i += L;
        }

        // fewer than `L` points remain: fall back to a scalar check for the remainder.
        (i..count).any(|i| {
            let mut distsq = A::ZERO;
            for (k, &c) in center.iter().enumerate() {
                let diff = voxel.axes[k][i] - c;
                distsq = distsq + diff.square();
            }
            distsq <= rsq
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mvt;
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    fn distsq<A: Axis, const K: usize>(a: [A; K], b: [A; K]) -> A {
        let mut total = A::ZERO;
        for k in 0..K {
            total = total + (a[k] - b[k]).square();
        }
        total
    }

    #[test]
    fn build_simple() {
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let t = MutableMvt::<2>::new(&points, 0.2);
        println!("{t:?}");
    }

    #[test]
    fn exact_query_single() {
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let t = MutableMvt::<2>::new(&points, 0.2);

        let q0 = [0.0, -0.01];
        assert!(t.collides(&q0, 0.12));
    }

    #[test]
    fn no_collision() {
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let t = MutableMvt::<2>::new(&points, 0.2);

        assert!(!t.collides(&[10.0, 10.0], 0.1));
    }

    #[test]
    fn three_d() {
        let points = [
            [0.0; 3],
            [0.1, -1.1, 0.5],
            [-0.2, -0.3, 0.25],
            [0.1, -1.1, 0.5],
        ];

        let t = MutableMvt::<3>::new(&points, 0.2);

        assert!(t.collides(&[0.0, 0.1, 0.0], 0.11));
        assert!(!t.collides(&[0.0, 0.1, 0.0], 0.05));
    }

    #[test]
    fn point_radius() {
        let points = [[0.0, 0.0], [0.0, 1.0]];
        let voxel_width = 1.0;

        let mvt = MutableMvt::<2>::with_point_radius(&points, voxel_width, 0.5);
        assert!(mvt.collides(&[0.6, 0.0], 0.2));
        assert!(!mvt.collides(&[0.6, 0.0], 0.05));
    }

    #[test]
    fn custom_index_type() {
        const R: f32 = 0.04;
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let mut rng = SmallRng::seed_from_u64(1234);
        let t: MutableMvt<2, f32, u16> = MutableMvt::new(&points, R);

        for _ in 0..10_000 {
            let p = [rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)];
            let collides = points.iter().any(|&a| distsq(a, p) <= R * R);
            assert_eq!(collides, t.collides(&p, R), "query point {p:?}");
        }
    }

    #[test]
    fn too_many_voxels_for_index_type() {
        // 300 points, each spaced far enough apart to land in its own voxel: more than `u8` (with
        // its top value reserved as a sentinel) can index.
        #[expect(
            clippy::cast_precision_loss,
            reason = "loop index is tiny relative to f32's mantissa"
        )]
        let points: Vec<[f32; 2]> = (0..300_i32).map(|i| [i as f32 * 10.0, 0.0]).collect();

        let result = MutableMvt::<2, f32, u8>::try_new(&points, 0.1);
        assert_eq!(result.unwrap_err(), NewMutableMvtError::TooManyVoxels);
    }

    #[test]
    fn non_cubic_workspace() {
        #[expect(
            clippy::cast_precision_loss,
            reason = "loop index is tiny relative to f32's mantissa"
        )]
        let points: Vec<[f32; 2]> = (0..200_i32).map(|i| [i as f32 * 0.5, 0.3]).collect();
        let t = MutableMvt::<2>::new(&points, 0.05);

        assert_eq!(
            t.grid_width[1], 1,
            "a single row suffices along the short axis"
        );
        assert!(t.grid_width[0] > t.grid_width[1]);

        assert!(t.collides(&[10.0, 0.3], 0.01));
        assert!(!t.collides(&[10.25, 0.3], 0.01));
        assert!(!t.collides(&[10.0, 10.0], 0.01));

        for &p in &points {
            let collides = points.iter().any(|&a| distsq(a, p) <= 0.05 * 0.05);
            assert_eq!(collides, t.collides(&p, 0.05), "query point {p:?}");
        }
    }

    #[test]
    fn empty_cloud_errors() {
        let points: [[f32; 2]; 0] = [];
        let err = MutableMvt::<2>::try_new(&points, 1.0).unwrap_err();
        assert_eq!(err, NewMutableMvtError::EmptyPointCloud);
    }

    #[test]
    #[should_panic(expected = "failed to construct MutableMvt")]
    fn empty_cloud_panics() {
        let points: [[f32; 2]; 0] = [];
        let _ = MutableMvt::<2>::new(&points, 1.0);
    }

    #[test]
    fn empty_cloud_via_with_workspace_never_collides() {
        // an empty `MutableMvt` built through `with_workspace` (rather than `new`'s point-cloud
        // path) must still behave correctly with no points inserted.
        let mvt = MutableMvt::<2>::with_workspace([0.0, 0.0], [1.0, 1.0], 1.0, 0.0);
        assert!(!mvt.collides(&[0.0, 0.0], 100.0));
    }

    #[test]
    fn single_point() {
        let points = [[1.0, 1.0]];
        let mvt = MutableMvt::<2>::new(&points, 1.0);
        assert!(mvt.collides(&[1.0, 1.0], 0.01));
        assert!(!mvt.collides(&[5.0, 5.0], 0.01));
    }

    #[test]
    fn get_points() {
        let mut points = [
            [-1.0, 0.0],
            [0.001, 0.0],
            [0.0, 0.5],
            [-1.0, 10.0],
            [-2.0, 10.0],
            [-0.5, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ];

        let mvt = MutableMvt::<2>::new(&points, 0.1);
        let mut points2 = mvt.points().collect::<Vec<_>>();

        points.sort_by(|a, b| a.partial_cmp(b).unwrap());
        points2.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(&points, &*points2);
    }

    #[test]
    fn fuzz() {
        const R: f32 = 0.04;
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let mut rng = SmallRng::seed_from_u64(1234);
        let t = MutableMvt::<2>::new(&points, R);

        for _ in 0..10_000 {
            let p = [rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)];
            let collides = points.iter().any(|&a| distsq(a, p) <= R * R);
            assert_eq!(collides, t.collides(&p, R), "query point {p:?}");
        }
    }

    #[test]
    fn fuzz_3d_dense() {
        const R: f32 = 0.3;
        let mut rng = SmallRng::seed_from_u64(42);
        let points: Vec<[f32; 3]> = (0..500)
            .map(|_| {
                [
                    rng.random_range(-5.0..5.0),
                    rng.random_range(-5.0..5.0),
                    rng.random_range(-5.0..5.0),
                ]
            })
            .collect();
        let t = MutableMvt::<3>::with_point_radius(&points, R, 0.05);

        for _ in 0..2_000 {
            let p = [
                rng.random_range(-6.0..6.0),
                rng.random_range(-6.0..6.0),
                rng.random_range(-6.0..6.0),
            ];
            let collides = points
                .iter()
                .any(|&a| distsq(a, p) <= (R + 0.05) * (R + 0.05));
            assert_eq!(collides, t.collides(&p, R), "query point {p:?}");
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn fuzz_simd_2d() {
        const R: f32 = 0.04;
        const L: usize = 8;
        let mut rng = SmallRng::seed_from_u64(7);
        let points: Vec<[f32; 2]> = (0..300)
            .map(|_| [rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)])
            .collect();
        let t = MutableMvt::<2>::new(&points, R);

        for _ in 0..2_000 {
            let batch: [[f32; L]; 2] =
                array::from_fn(|_| array::from_fn(|_| rng.random_range(-1.5..1.5)));
            let centers = batch.map(Simd::from_array);
            let radii = Simd::splat(R);

            let expected = (0..L).any(|lane| {
                let p = [batch[0][lane], batch[1][lane]];
                points.iter().any(|&a| distsq(a, p) <= R * R)
            });
            assert_eq!(
                expected,
                t.collides_simd(&centers, radii),
                "batch {batch:?}"
            );
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn fuzz_simd_3d_with_point_radius() {
        const R: f32 = 0.3;
        const R_POINT: f32 = 0.05;
        const L: usize = 4;
        let mut rng = SmallRng::seed_from_u64(99);
        let points: Vec<[f32; 3]> = (0..400)
            .map(|_| {
                [
                    rng.random_range(-5.0..5.0),
                    rng.random_range(-5.0..5.0),
                    rng.random_range(-5.0..5.0),
                ]
            })
            .collect();
        let t = MutableMvt::<3>::with_point_radius(&points, R, R_POINT);

        for _ in 0..1_000 {
            let batch: [[f32; L]; 3] =
                array::from_fn(|_| array::from_fn(|_| rng.random_range(-6.0..6.0)));
            let centers = batch.map(Simd::from_array);
            let radii = Simd::splat(R);

            let expected = (0..L).any(|lane| {
                let p = [batch[0][lane], batch[1][lane], batch[2][lane]];
                points
                    .iter()
                    .any(|&a| distsq(a, p) <= (R + R_POINT) * (R + R_POINT))
            });
            assert_eq!(
                expected,
                t.collides_simd(&centers, radii),
                "batch {batch:?}"
            );
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn simd_matches_scalar_exact_hit() {
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let t = MutableMvt::<2>::new(&points, 0.2);

        // only the first lane is on a colliding query; the rest are far away.
        let centers = [
            Simd::from_array([0.0, 10.0, -10.0, 5.0]),
            Simd::from_array([-0.01, 10.0, -10.0, 5.0]),
        ];
        let radii = Simd::splat(0.12);

        assert!(t.collides_simd(&centers, radii));
        assert!(t.collides(&[0.0, -0.01], 0.12));
    }

    #[cfg(feature = "simd")]
    #[test]
    fn simd_no_collision() {
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let t = MutableMvt::<2>::new(&points, 0.2);

        let centers = [
            Simd::from_array([10.0, -10.0, 20.0, -20.0]),
            Simd::from_array([10.0, -10.0, 20.0, -20.0]),
        ];
        let radii = Simd::splat(0.1);

        assert!(!t.collides_simd(&centers, radii));
    }

    // ==== insertion-specific tests ====

    #[test]
    fn insert_matches_build_all_at_once() {
        const R: f32 = 0.05;
        let mut rng = SmallRng::seed_from_u64(2024);
        let points: Vec<[f32; 3]> = (0..400)
            .map(|_| {
                [
                    rng.random_range(-3.0..3.0),
                    rng.random_range(-3.0..3.0),
                    rng.random_range(-3.0..3.0),
                ]
            })
            .collect();

        // build a reference structure with every point at once...
        let built_all_at_once = Mvt::<3, f32>::new(&points, R);

        // ...and an equivalent one by inserting one point at a time into a structure seeded with
        // just the first point, exercising the same workspace bounds either way.
        let mut inserted_one_at_a_time = MutableMvt::<3, f32>::new(&points[..1], R);
        for p in &points[1..] {
            inserted_one_at_a_time.insert(p).unwrap();
        }

        for _ in 0..2_000 {
            let center = [
                rng.random_range(-3.5..3.5),
                rng.random_range(-3.5..3.5),
                rng.random_range(-3.5..3.5),
            ];
            let radius = rng.random_range(0.0..R);
            assert_eq!(
                built_all_at_once.collides(&center, radius),
                inserted_one_at_a_time.collides(&center, radius),
                "query {center:?}, radius {radius}"
            );
        }
    }

    #[test]
    fn insert_points_batch_matches_brute_force() {
        const R: f32 = 0.05;
        let mut rng = SmallRng::seed_from_u64(7331);
        let initial: Vec<[f32; 2]> = (0..50)
            .map(|_| [rng.random_range(-2.0..2.0), rng.random_range(-2.0..2.0)])
            .collect();
        let inserted: Vec<[f32; 2]> = (0..50)
            .map(|_| [rng.random_range(-2.0..2.0), rng.random_range(-2.0..2.0)])
            .collect();

        let mut mvt = MutableMvt::<2>::new(&initial, R);
        mvt.insert_points(&inserted).unwrap();

        let all_points: Vec<[f32; 2]> = initial.iter().chain(&inserted).copied().collect();

        for _ in 0..2_000 {
            let center = [rng.random_range(-2.5..2.5), rng.random_range(-2.5..2.5)];
            let radius = rng.random_range(0.0..R);
            let expected = all_points
                .iter()
                .any(|&a| distsq(a, center) <= radius * radius);
            assert_eq!(
                expected,
                mvt.collides(&center, radius),
                "query {center:?}, radius {radius}"
            );
        }
    }

    #[test]
    fn insert_rejects_non_finite() {
        let mut mvt = MutableMvt::<2>::new(&[[0.0, 0.0]], 1.0);
        let err = mvt.insert(&[f32::NAN, 0.0]).unwrap_err();
        assert_eq!(err, InsertError::NonFinite);
        // the rejected point must not have been stored.
        assert_eq!(mvt.points().count(), 1);
    }

    #[test]
    fn with_workspace_matches_build_from_points() {
        const R: f32 = 0.05;
        let mut rng = SmallRng::seed_from_u64(11);
        let points: Vec<[f32; 2]> = (0..300)
            .map(|_| [rng.random_range(-2.0..2.0), rng.random_range(-2.0..2.0)])
            .collect();

        let mut mvt = MutableMvt::<2>::with_workspace([-2.0, -2.0], [2.0, 2.0], R, 0.0);
        assert!(mvt.points().next().is_none());
        mvt.insert_points(&points).unwrap();

        for _ in 0..2_000 {
            let center = [rng.random_range(-2.5..2.5), rng.random_range(-2.5..2.5)];
            let radius = rng.random_range(0.0..R);
            let expected = points.iter().any(|&a| distsq(a, center) <= radius * radius);
            assert_eq!(
                expected,
                mvt.collides(&center, radius),
                "query {center:?}, radius {radius}"
            );
        }
    }

    #[test]
    fn with_workspace_rejects_non_finite() {
        let err =
            MutableMvt::<2>::try_with_workspace([0.0, f32::NAN], [1.0, 1.0], 0.1, 0.0).unwrap_err();
        assert_eq!(err, NewMutableMvtError::NonFinite);
    }

    #[test]
    fn with_workspace_rejects_invalid_radius() {
        let err =
            MutableMvt::<2>::try_with_workspace([0.0, 0.0], [1.0, 1.0], -1.0, 0.0).unwrap_err();
        assert_eq!(err, NewMutableMvtError::InvalidVoxelWidth);
    }

    #[test]
    fn with_workspace_voxel_width_independent_of_point_radius() {
        let mvt = MutableMvt::<2>::try_with_workspace([0.0, 0.0], [1.0, 1.0], 1.0, -2.0).unwrap();
        assert!(mvt.points().next().is_none());
    }

    #[test]
    fn with_workspace_rejects_inverted_bounds() {
        let err =
            MutableMvt::<2>::try_with_workspace([1.0, 0.0], [0.0, 1.0], 0.1, 0.0).unwrap_err();
        assert_eq!(err, NewMutableMvtError::InvalidWorkspace);
    }

    #[test]
    fn with_workspace_point_radius_matches_build_from_points() {
        const R: f32 = 0.3;
        const R_POINT: f32 = 0.05;
        let mut rng = SmallRng::seed_from_u64(23);
        let points: Vec<[f32; 2]> = (0..300)
            .map(|_| [rng.random_range(-2.0..2.0), rng.random_range(-2.0..2.0)])
            .collect();

        let mut mvt = MutableMvt::<2>::with_workspace([-2.0, -2.0], [2.0, 2.0], R, R_POINT);
        mvt.insert_points(&points).unwrap();

        for _ in 0..2_000 {
            let center = [rng.random_range(-2.5..2.5), rng.random_range(-2.5..2.5)];
            let radius = rng.random_range(0.0..R);
            let expected = points
                .iter()
                .any(|&a| distsq(a, center) <= (radius + R_POINT) * (radius + R_POINT));
            assert_eq!(
                expected,
                mvt.collides(&center, radius),
                "query {center:?}, radius {radius}"
            );
        }
    }

    #[test]
    fn out_of_workspace_insert_still_correct() {
        // establish a tiny workspace from a tight cluster...
        let points = [[0.0, 0.0], [0.05, 0.05], [-0.05, 0.02]];
        let mut mvt = MutableMvt::<2>::new(&points, 0.1);

        // ...then insert a point far outside it. Its grid bucket will be clamped to an edge
        // voxel, but its true coordinates must still be stored and found exactly.
        let far_point = [1000.0, -1000.0];
        mvt.insert(&far_point).unwrap();

        assert!(mvt.collides(&far_point, 0.01));
        assert!(!mvt.collides(&[999.0, -1000.0], 0.01));

        // the original cluster must still query correctly too.
        assert!(mvt.collides(&[0.0, 0.0], 0.01));
        assert!(!mvt.collides(&[5.0, 5.0], 0.01));
    }

    #[cfg(feature = "simd")]
    #[test]
    fn insert_then_simd_query_matches_scalar() {
        const R: f32 = 0.3;
        const L: usize = 4;
        let mut rng = SmallRng::seed_from_u64(555);
        let initial: Vec<[f32; 3]> = (0..100)
            .map(|_| {
                [
                    rng.random_range(-3.0..3.0),
                    rng.random_range(-3.0..3.0),
                    rng.random_range(-3.0..3.0),
                ]
            })
            .collect();
        let mut mvt = MutableMvt::<3>::new(&initial, R);
        let inserted: Vec<[f32; 3]> = (0..100)
            .map(|_| {
                [
                    rng.random_range(-3.0..3.0),
                    rng.random_range(-3.0..3.0),
                    rng.random_range(-3.0..3.0),
                ]
            })
            .collect();
        mvt.insert_points(&inserted).unwrap();

        for _ in 0..500 {
            let batch: [[f32; L]; 3] =
                array::from_fn(|_| array::from_fn(|_| rng.random_range(-3.5..3.5)));
            let centers = batch.map(Simd::from_array);
            let radii = Simd::splat(R);

            let expected = (0..L).any(|lane| {
                let p = [batch[0][lane], batch[1][lane], batch[2][lane]];
                mvt.collides(&p, R)
            });
            assert_eq!(
                expected,
                mvt.collides_simd(&centers, radii),
                "batch {batch:?}"
            );
        }
    }
}
