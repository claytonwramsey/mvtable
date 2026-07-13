//! This is a Rust implementation of the *multilevel voxel table* (MVT), a data structure
//! for fast collision checking between spheres and
//! point clouds.
//!
//! If you use this in an academic work, please cite it as follows:
//!
//! ```bibtex
//! @inproceedings{chen2026vcc,
//!  author    = {Ching Chen and Tsung-Tai Yeh},
//!  title     = {VCC: Efficient Voxel-Based Collision Checking Framework for Real-Time Robotic
//!               Motion Planning},
//!  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
//!  year      = {2026},
//! }
//! ```
//!
//! ## Usage
//!
//! The core data structure in this library is the [`Mvt`], a sparse voxel grid used for
//! collision checking. [`Mvt`]s are polymorphic over dimension and floating-point type. On
//! construction, they take in a list of points in a point cloud and the maximum radius that will
//! be used for querying, which is used to size the grid's voxels.
//!
//! ```rust
//! use mvtable::Mvt;
//!
//! // list of points in cloud
//! let points = [[0.0, 1.1], [0.2, 3.1]];
//! let r_max = 2.0;
//!
//! let mvt = Mvt::<2>::new(&points, r_max);
//! ```
//!
//! Once you have an [`Mvt`], you can use it for collision-checking against spheres.
//!
//! ```rust
//! # use mvtable::Mvt;
//! # let points = [[0.0, 1.1], [0.2, 3.1]];
//! # let mvt = Mvt::<2>::new(&points, 2.0);
//! let center = [0.0, 0.0]; // center of sphere
//! let radius0 = 1.0; // radius of sphere
//! assert!(!mvt.collides(&center, radius0));
//!
//! let radius1 = 1.5;
//! assert!(mvt.collides(&center, radius1));
//! ```
//!
//! [`Mvt`] is immutable once built. If you need to insert new points after construction, use
//! [`MutableMvt`] instead, which supports [`MutableMvt::insert`]/[`MutableMvt::insert_points`] at
//! some cost to query performance.
//!
//! ## Optional features
//!
//! This crate exposes one feature, `simd`, which enables a SIMD-parallel interface for querying
//! [`Mvt`]s. The `simd` feature requires nightly Rust and therefore should be considered
//! unstable. This enables the function `Mvt::collides_simd`, a parallel collision checker for
//! batches of search queries.
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "simd", feature(portable_simd))]
#![warn(clippy::pedantic, clippy::nursery)]
#![warn(clippy::allow_attributes, reason = "prefer expect over allow")]
#![cfg_attr(doc, feature(rustdoc_missing_doc_code_examples))]
#![warn(missing_docs, rustdoc::missing_doc_code_examples)]

extern crate alloc;

use alloc::{boxed::Box, vec, vec::Vec};
use core::{
    array, fmt,
    mem::size_of,
    ops::{Add, Div, Mul, Sub},
};

mod grid;
mod mutable;

pub use mutable::{InsertError, MutableMvt, NewMutableMvtError};

#[cfg(feature = "simd")]
use core::ops::AddAssign;
#[cfg(feature = "simd")]
use core::simd::{
    Select, Simd, SimdElement,
    cmp::{SimdPartialEq, SimdPartialOrd},
};

/// A generic trait representing values that may be used as an axis; that is, elements of a
/// vector representing a point.
///
/// An array of `Axis` values is a point that can be stored in an [`Mvt`]. This trait is
/// implemented for `f32` and `f64`.
///
/// # Examples
///
/// ```
/// use mvtable::Axis;
///
/// assert_eq!(f32::ZERO, 0.0);
/// assert!(!f32::INFINITY.is_finite());
/// assert_eq!(2.0_f32.square(), 4.0);
/// ```
pub trait Axis:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
{
    /// A zero value.
    const ZERO: Self;
    /// A value that is larger than any finite value.
    const INFINITY: Self;
    /// A value that is smaller than any finite value.
    const NEG_INFINITY: Self;

    #[must_use]
    #[expect(rustdoc::missing_doc_code_examples)]
    /// Determine whether this value is finite.
    fn is_finite(self) -> bool;

    #[must_use]
    #[expect(rustdoc::missing_doc_code_examples)]
    /// Compute the square of this value.
    fn square(self) -> Self;

    #[must_use]
    /// Convert a non-negative grid coordinate to an index, truncating any fractional part.
    ///
    /// Values less than zero saturate to `0`, and values that are too large to be represented
    /// saturate to [`usize::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use mvtable::Axis;
    ///
    /// assert_eq!(3.7_f32.to_index(), 3);
    /// assert_eq!((-1.0_f32).to_index(), 0);
    /// ```
    fn to_index(self) -> usize;

    #[must_use]
    #[expect(rustdoc::missing_doc_code_examples)]
    /// Convert a grid width into an axis value.
    fn from_usize(x: usize) -> Self;
}

#[cfg(feature = "simd")]
#[expect(rustdoc::missing_doc_code_examples)]
/// A trait used for SIMD elements, implemented for the same types that implement [`Axis`].
///
/// This trait (and [`Mvt::collides_simd`], which requires it) is only available when the `simd`
/// feature is enabled, which requires a nightly compiler.
pub trait AxisSimdElement: SimdElement + Default + Axis {}

#[cfg(feature = "simd")]
/// A trait used for masks over SIMD vectors of [`Axis`] values, used for parallel querying on
/// [`Mvt`]s.
///
/// The interface for this trait should be considered unstable since the standard SIMD API may
/// change with Rust versions.
///
/// # Examples
///
/// ```
/// #![feature(portable_simd)]
/// use std::simd::{Simd, cmp::SimdPartialEq};
///
/// use mvtable::AxisSimd;
///
/// let a = Simd::from_array([1.0f32, 2.0, 3.0, 4.0]);
/// let mask = a.simd_eq(Simd::splat(2.0));
/// assert!(Simd::<f32, 4>::mask_any(mask));
/// ```
pub trait AxisSimd<const L: usize>:
    Sized + SimdPartialOrd + Add<Output = Self> + AddAssign + Sub<Output = Self> + Mul<Output = Self>
{
    #[must_use]
    #[expect(rustdoc::missing_doc_code_examples)]
    /// Determine whether a mask contains any true elements.
    fn mask_any(mask: <Self as SimdPartialEq>::Mask) -> bool;

    #[must_use]
    #[expect(rustdoc::missing_doc_code_examples)]
    /// Choose, lane by lane, between `true_val` and `false_val` according to `mask`.
    fn select(mask: <Self as SimdPartialEq>::Mask, true_val: Self, false_val: Self) -> Self;

    #[must_use]
    #[expect(rustdoc::missing_doc_code_examples)]
    /// Convert a mask into a per-lane array of `bool`s.
    fn mask_to_array(mask: <Self as SimdPartialEq>::Mask) -> [bool; L];
}

macro_rules! impl_axis {
    ($t: ty) => {
        impl Axis for $t {
            const ZERO: Self = 0.0;
            const INFINITY: Self = <$t>::INFINITY;
            const NEG_INFINITY: Self = <$t>::NEG_INFINITY;

            fn is_finite(self) -> bool {
                <$t>::is_finite(self)
            }

            fn square(self) -> Self {
                self * self
            }

            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "saturating float-to-int cast is exactly the desired clamping behavior"
            )]
            fn to_index(self) -> usize {
                self as usize
            }

            #[expect(
                clippy::cast_precision_loss,
                reason = "grid widths are small enough to be represented exactly as floats"
            )]
            fn from_usize(x: usize) -> Self {
                x as $t
            }
        }

        #[cfg(feature = "simd")]
        impl AxisSimdElement for $t {}

        #[cfg(feature = "simd")]
        impl<const L: usize> AxisSimd<L> for Simd<$t, L> {
            fn mask_any(mask: <Self as SimdPartialEq>::Mask) -> bool {
                mask.any()
            }

            fn select(
                mask: <Self as SimdPartialEq>::Mask,
                true_val: Self,
                false_val: Self,
            ) -> Self {
                mask.select(true_val, false_val)
            }

            fn mask_to_array(mask: <Self as SimdPartialEq>::Mask) -> [bool; L] {
                mask.to_array()
            }
        }
    };
}

impl_axis!(f32);
impl_axis!(f64);

/// An integer type used to address entries in the table pool and voxel array.
///
/// This is implemented so that [`Mvt`]s can use smaller index types (such as [`u16`] or [`u32`])
/// for improved memory density, at the cost of supporting fewer voxels and points. This trait is
/// implemented for [`u8`], [`u16`], [`u32`], [`u64`], and [`usize`].
///
/// # Examples
///
/// ```
/// use mvtable::Index;
///
/// assert_eq!(u32::from_usize(5), Some(5));
/// assert_eq!(5u32.to_usize(), 5);
///
/// // `u8` can't represent every `usize`, so out-of-range values convert to `None`.
/// assert_eq!(u8::from_usize(1_000), None);
/// ```
pub trait Index: Copy + PartialEq {
    /// The zero index.
    const ZERO: Self;
    /// The sentinel value used to mark an empty (unallocated) table entry. An index equal to
    /// this value can never be produced by [`Index::from_usize`].
    const SENTINEL: Self;

    #[must_use]
    #[expect(rustdoc::missing_doc_code_examples)]
    /// Convert a `usize` into an index, or `None` if it doesn't fit (or happens to equal
    /// [`Index::SENTINEL`]).
    fn from_usize(x: usize) -> Option<Self>;

    #[must_use]
    #[expect(rustdoc::missing_doc_code_examples)]
    /// Convert this index back into a `usize`.
    fn to_usize(self) -> usize;
}

macro_rules! impl_index {
    ($t: ty) => {
        impl Index for $t {
            const ZERO: Self = 0;
            const SENTINEL: Self = <$t>::MAX;

            fn from_usize(x: usize) -> Option<Self> {
                let v = Self::try_from(x).ok()?;
                (v != Self::SENTINEL).then_some(v)
            }

            fn to_usize(self) -> usize {
                self as usize
            }
        }
    };
}

impl_index!(u8);
impl_index!(u16);
impl_index!(u32);
impl_index!(usize);

// special case to suppress warnings
impl Index for u64 {
    const ZERO: Self = 0;
    const SENTINEL: Self = Self::MAX;
    fn from_usize(x: usize) -> Option<Self> {
        let v = Self::try_from(x).ok()?;
        (v != Self::SENTINEL).then_some(v)
    }
    #[expect(clippy::cast_possible_truncation)]
    fn to_usize(self) -> usize {
        self as usize
    }
}

/// An axis-aligned bounding box, used both as a global bound on the point cloud and as a local
/// bound on the points contained by a single voxel.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Aabb<A, const K: usize> {
    lo: [A; K],
    hi: [A; K],
}

impl<A: Axis, const K: usize> Aabb<A, K> {
    /// A bounding box that contains no points; inserting any point will grow it to contain
    /// exactly that point.
    const EMPTY: Self = Self {
        lo: [A::INFINITY; K],
        hi: [A::NEG_INFINITY; K],
    };

    /// Grow this bounding box so that it also contains `p`.
    fn insert(&mut self, p: &[A; K]) {
        for ((l, h), &x) in self.lo.iter_mut().zip(&mut self.hi).zip(p) {
            if x < *l {
                *l = x;
            }
            if x > *h {
                *h = x;
            }
        }
    }

    /// Compute the squared distance from `p` to the closest point contained by this box.
    fn closest_distsq_to(&self, p: &[A; K]) -> A {
        let mut total = A::ZERO;
        for ((&lo, &hi), &x) in self.lo.iter().zip(&self.hi).zip(p) {
            let clamped = if x < lo {
                lo
            } else if x > hi {
                hi
            } else {
                x
            };
            total = total + (x - clamped).square();
        }
        total
    }

    /// Compute the component-wise bounding box over `points`, or `None` if `points` is empty.
    fn bounding_box(points: &[[A; K]]) -> Option<Self> {
        let (first, rest) = points.split_first()?;
        let mut lo = *first;
        let mut hi = *first;
        for p in rest {
            for k in 0..K {
                if p[k] < lo[k] {
                    lo[k] = p[k];
                }
                if p[k] > hi[k] {
                    hi[k] = p[k];
                }
            }
        }
        Some(Self { lo, hi })
    }
}

/// Metadata for a single occupied voxel.
#[derive(Clone, Copy, Debug)]
struct Voxel<A, I, const K: usize> {
    /// A local bounding box over the points contained by this voxel.
    aabb: Aabb<A, K>,
    /// The offset of this voxel's points within the point coordinate pool.
    offset: I,
    /// The number of points contained by this voxel.
    count: I,
}

/// The intermediate result of [`Mvt::build_hierarchy`]: the table pool, together with the points
/// and bounding box accumulated so far for each voxel encountered, in first-encounter order.
type VoxelBuckets<A, I, const K: usize> = (Vec<I>, Vec<Vec<[A; K]>>, Vec<Aabb<A, K>>);

/// The result of [`Mvt::flatten_points`]: metadata for each voxel, together with the point
/// coordinate pool.
type FlattenedVoxels<A, I, const K: usize> = (Vec<Voxel<A, I, K>>, Vec<A>);

#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
/// The errors that can occur when calling [`Mvt::try_new`] or [`Mvt::try_with_point_radius`].
///
/// This type is specific to constructing an [`Mvt`]; [`MutableMvt`] has its
/// own, separate error types for construction
/// ([`NewMutableMvtError`]) and insertion
/// ([`InsertError`]), since (for example) an `Mvt` can never fail to
/// construct in the specific way an uninitialized `MutableMvt` can fail to accept an insertion.
///
/// # Examples
///
/// ```
/// let points = [[0.0]];
/// let err = mvtable::Mvt::<1>::try_new(&points, -1.0).unwrap_err();
/// assert_eq!(err, mvtable::NewMvtError::InvalidRadius);
/// ```
pub enum NewMvtError {
    /// At least one of the points had a non-finite value.
    NonFinite,
    /// The combined radius (`r_max + r_point`) was not a positive, finite value, so voxels could
    /// not be sized.
    InvalidRadius,
    /// There were too many voxels or points to be represented without integer overflow.
    TooManyVoxels,
}

impl From<grid::TooManyVoxels> for NewMvtError {
    fn from(_: grid::TooManyVoxels) -> Self {
        Self::TooManyVoxels
    }
}

impl fmt::Display for NewMvtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => write!(f, "at least one point had a non-finite value"),
            Self::InvalidRadius => {
                write!(f, "r_max + r_point was not a positive, finite value")
            }
            Self::TooManyVoxels => {
                write!(
                    f,
                    "too many voxels or points for the index type to represent"
                )
            }
        }
    }
}

impl core::error::Error for NewMvtError {}

#[derive(Clone, Debug)]
/// A multilevel voxel tree, a structure for point cloud collision checking.
///
/// # Generic parameters
///
/// - `K`: The dimension of the space.
/// - `A`: The value of the axes of each point. This should typically be `f32` or `f64`. This should
///   implement [`Axis`].
/// - `I`: The index integer used internally to address voxels and points. This should generally be
///   an unsigned integer type, such as `u32` or `usize`. This should implement [`Index`].
///
/// # Citation
///
/// ```bibtex
/// @inproceedings{chen2026vcc,
///  author    = {Ching Chen and Tsung-Tai Yeh},
///  title     = {VCC: Efficient Voxel-Based Collision Checking Framework for Real-Time Robotic Motion Planning},
///  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
///  year      = {2026},
/// }
/// ```
///
/// # Examples
///
/// ```rust
/// // list of points in cloud
/// let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
///
/// // query radii must be between 0.0 and 0.2
/// let t = mvtable::Mvt::<2>::new(&points, 0.2);
///
/// assert!(!t.collides(&[0.0, 0.3], 0.1));
/// assert!(t.collides(&[0.0, 0.2], 0.15));
/// ```
pub struct Mvt<const K: usize, A = f32, I = u32> {
    /// The number of voxels along each axis of the grid. Axes need not have the same number of
    /// voxels, so the workspace need not be cubic.
    grid_width: [I; K],
    /// The number of grid cells per unit length along each axis, i.e.
    /// `grid_width[k] / workspace_width[k]`.
    scale: [A; K],
    /// The radius to add to every point to account for its physical volume.
    r_point: A,
    /// A bounding box over every point in the cloud, used to quickly reject far-away queries.
    global_aabb: Aabb<A, K>,
    /// The table pool: the concatenation of the root table and every subsequently allocated
    /// table, storing offsets into this same pool for the first `K - 1` levels, and voxel
    /// indices (into `voxels`) for the last level. Empty entries are marked with
    /// [`Index::SENTINEL`].
    tables: Box<[I]>,
    /// Metadata (bounding box, and location within `points`) for each occupied voxel.
    voxels: Box<[Voxel<A, I, K>]>,
    /// The point coordinate pool: for each voxel (in the order they appear in `voxels`), the
    /// coordinates of its points stored in struct-of-arrays order, i.e. all the 0th coordinates,
    /// then all the 1st coordinates, and so on.
    points: Box<[A]>,
}

impl<const K: usize, A: Axis, I: Index> Mvt<K, A, I> {
    /// Construct a new MVT containing all the points in `points`.
    ///
    /// `r_max` is the maximum radius of the balls that will be queried against the tree; it is
    /// used to size the grid's voxels.
    ///
    /// # Panics
    ///
    /// This function will panic if any point contains a non-finite value, or if `r_max` is not a
    /// positive, finite value.
    ///
    /// # Examples
    ///
    /// ```
    /// let points = [[0.0]];
    ///
    /// let mvt = mvtable::Mvt::<1>::new(&points, f32::INFINITY);
    ///
    /// assert!(mvt.collides(&[1.0], 1.5));
    /// assert!(!mvt.collides(&[1.0], 0.5));
    /// ```
    pub fn new(points: &[[A; K]], r_max: A) -> Self {
        Self::try_new(points, r_max).expect("failed to construct Mvt; see NewMvtError variants")
    }

    /// Construct a new MVT containing all the points in `points`, with a point radius `r_point`
    /// added to every query.
    ///
    /// # Panics
    ///
    /// This function will panic if any point contains a non-finite value, or if
    /// `r_max + r_point` is not a positive, finite value.
    ///
    /// # Examples
    ///
    /// ```
    /// let points = [[0.0]];
    ///
    /// let mvt = mvtable::Mvt::<1>::with_point_radius(&points, f32::INFINITY, 0.2);
    ///
    /// assert!(mvt.collides(&[1.0], 1.5));
    /// assert!(!mvt.collides(&[1.0], 0.5));
    /// ```
    pub fn with_point_radius(points: &[[A; K]], r_max: A, r_point: A) -> Self {
        Self::try_with_point_radius(points, r_max, r_point)
            .expect("failed to construct Mvt; see NewMvtError variants")
    }

    /// Construct a new MVT containing all the points in `points`, checking for invalid input.
    ///
    /// # Errors
    ///
    /// See [`NewMvtError`] for the circumstances in which this function returns an error.
    ///
    /// # Examples
    ///
    /// ```
    /// let points = [[0.0]];
    /// let mvt = mvtable::Mvt::<1>::try_new(&points, f32::INFINITY)?;
    /// # Ok::<(), mvtable::NewMvtError>(())
    /// ```
    pub fn try_new(points: &[[A; K]], r_max: A) -> Result<Self, NewMvtError> {
        Self::try_with_point_radius(points, r_max, A::ZERO)
    }

    /// Construct a new MVT containing all the points in `points`, with a point radius `r_point`
    /// added to every query, checking for invalid input.
    ///
    /// # Errors
    ///
    /// See [`NewMvtError`] for the circumstances in which this function returns an error.
    ///
    /// # Examples
    ///
    /// ```
    /// let points = [[0.0]];
    /// let mvt = mvtable::Mvt::<1>::try_with_point_radius(&points, f32::INFINITY, 0.01)?;
    /// # Ok::<(), mvtable::NewMvtError>(())
    /// ```
    pub fn try_with_point_radius(
        points: &[[A; K]],
        r_max: A,
        r_point: A,
    ) -> Result<Self, NewMvtError> {
        const { assert!(K > 0, "Mvt requires at least one dimension") };

        if points.iter().any(|p| p.iter().any(|x| !x.is_finite())) {
            return Err(NewMvtError::NonFinite);
        }
        let cell_wd = r_max + r_point;
        if cell_wd <= A::ZERO {
            return Err(NewMvtError::InvalidRadius);
        }

        let Some(global_aabb) = Aabb::bounding_box(points) else {
            // no points: return an empty MVT that never collides.
            return Ok(Self {
                grid_width: [I::ZERO; K],
                scale: [A::ZERO; K],
                r_point,
                global_aabb: Aabb::EMPTY,
                tables: Box::default(),
                voxels: Box::default(),
                points: Box::default(),
            });
        };

        let (grid_width, grid_width_i, scale) = grid::size_grid(&global_aabb, cell_wd)?;

        let (tables, voxel_points, voxel_aabbs) =
            Self::build_hierarchy(points, global_aabb.lo, scale, grid_width)?;
        let (voxels, pool) = Self::flatten_points(voxel_points, voxel_aabbs)?;

        Ok(Self {
            grid_width: grid_width_i,
            scale,
            r_point,
            global_aabb,
            tables: tables.into_boxed_slice(),
            voxels: voxels.into_boxed_slice(),
            points: pool.into_boxed_slice(),
        })
    }

    /// Phase 1 of construction: build the sparse table hierarchy and bucket points into
    /// per-voxel accumulators, indexed in the same order the voxels were first encountered.
    ///
    /// Level `level` of the hierarchy is indexed by grid coordinates along axis `level`, so a
    /// table for level `level` always has `grid_width[level]` entries.
    fn build_hierarchy(
        points: &[[A; K]],
        lo: [A; K],
        scale: [A; K],
        grid_width: [usize; K],
    ) -> Result<VoxelBuckets<A, I, K>, NewMvtError> {
        let mut tables: Vec<I> = grid::new_root_table(grid_width);
        let mut voxel_points: Vec<Vec<[A; K]>> = Vec::new();
        let mut voxel_aabbs: Vec<Aabb<A, K>> = Vec::new();

        for p in points {
            let coords = grid::point_to_grid_coords(p, lo, scale, grid_width);
            let leaf_slot = grid::get_leaf(&mut tables, grid_width, coords)?;

            let voxel_idx = if tables[leaf_slot] == I::SENTINEL {
                let idx = voxel_points.len();
                voxel_points.push(Vec::new());
                voxel_aabbs.push(Aabb::EMPTY);
                tables[leaf_slot] = I::from_usize(idx).ok_or(NewMvtError::TooManyVoxels)?;
                idx
            } else {
                tables[leaf_slot].to_usize()
            };

            voxel_points[voxel_idx].push(*p);
            voxel_aabbs[voxel_idx].insert(p);
        }

        Ok((tables, voxel_points, voxel_aabbs))
    }

    /// Phase 2 of construction: flatten the per-voxel point buffers built by
    /// [`Self::build_hierarchy`] into a single struct-of-arrays pool.
    fn flatten_points(
        voxel_points: Vec<Vec<[A; K]>>,
        voxel_aabbs: Vec<Aabb<A, K>>,
    ) -> Result<FlattenedVoxels<A, I, K>, NewMvtError> {
        let total_points: usize = voxel_points.iter().map(Vec::len).sum();
        let mut pool = vec![A::ZERO; total_points * K];
        let mut voxels = Vec::with_capacity(voxel_points.len());
        let mut offset = 0usize;
        for (pts, aabb) in voxel_points.into_iter().zip(voxel_aabbs) {
            let count = pts.len();
            for (k, coord_pool) in pool[offset..].chunks_mut(count).take(K).enumerate() {
                for (dst, p) in coord_pool.iter_mut().zip(&pts) {
                    *dst = p[k];
                }
            }
            voxels.push(Voxel {
                aabb,
                offset: I::from_usize(offset).ok_or(NewMvtError::TooManyVoxels)?,
                count: I::from_usize(count).ok_or(NewMvtError::TooManyVoxels)?,
            });
            offset += count * K;
        }

        Ok((voxels, pool))
    }

    /// Look up the voxel containing grid coordinates `coords`, if it is occupied.
    fn lookup_voxel(&self, coords: &[usize; K]) -> Option<&Voxel<A, I, K>> {
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
    /// let mvt = mvtable::Mvt::<3>::new(&points, 0.2);
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
            let base = voxel.offset.to_usize();
            let count = voxel.count.to_usize();
            (0..count).any(|i| {
                let mut distsq = A::ZERO;
                for (k, &c) in center.iter().enumerate() {
                    let diff = self.points[base + k * count + i] - c;
                    distsq = distsq + diff.square();
                }
                distsq <= rsq
            })
        })
    }

    /// Search the block of voxels that could contain a point within `r` (already including
    /// `r_point`) of `center`, calling `check_voxel` on each voxel whose local AABB could contain
    /// such a point. Returns `true` as soon as `check_voxel` does, and `false` if every voxel in
    /// the block has been checked without one returning `true`.
    ///
    /// The caller is responsible for having already checked that the query sphere intersects the
    /// global bounding box over the whole point cloud; this function does not repeat that check,
    /// since a batched SIMD caller may have already performed an equivalent check for every lane
    /// in the batch at once.
    fn search_block(
        &self,
        center: &[A; K],
        r: A,
        check_voxel: impl Fn(&Voxel<A, I, K>) -> bool,
    ) -> bool {
        let rsq = r.square();

        let mut bmin = [0usize; K];
        let mut bmax = [0usize; K];
        for k in 0..K {
            let grid_max = self.grid_width[k].to_usize() - 1;
            // theoretically has epsilon-scale errors, but is ok
            let rg = r * self.scale[k];
            let v = (center[k] - self.global_aabb.lo[k]) * self.scale[k];
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

    /// Get an iterator over the points stored in this `Mvt`.
    /// It makes no guarantee of iteration order.
    ///
    /// ```
    /// let mvt = mvtable::Mvt::<2>::new(&[[0.0, 1.0]], f32::INFINITY);
    /// for point in mvt.points() {
    ///     println!("{point:?}");
    /// }
    /// ```
    pub fn points(&self) -> impl Iterator<Item = [A; K]> + '_ {
        self.voxels.iter().flat_map(move |v| {
            let base = v.offset.to_usize();
            let count = v.count.to_usize();
            (0..count).map(move |i| array::from_fn(|k| self.points[base + k * count + i]))
        })
    }

    #[must_use]
    #[doc(hidden)]
    /// Get the total memory used (stack + heap) by this structure, measured in bytes.
    /// This function should not be considered stable; it is only used internally for benchmarks.
    pub fn memory_used(&self) -> usize {
        size_of::<Self>()
            + self.tables.len() * size_of::<I>()
            + self.voxels.len() * size_of::<Voxel<A, I, K>>()
            + self.points.len() * size_of::<A>()
    }
}

#[cfg(feature = "simd")]
impl<const K: usize, A: AxisSimdElement, I: Index> Mvt<K, A, I> {
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
    /// let mvt = mvtable::Mvt::<2>::new(&points, 0.1);
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
                let base = voxel.offset.to_usize();
                let count = voxel.count.to_usize();
                self.points_collide_simd::<L>(base, count, &center, rsq_lane)
            })
        })
    }

    /// Determine whether any of the `count` points stored at `base` in the point coordinate pool
    /// are within a squared distance of `rsq` from `center`, checking `L` points at a time.
    fn points_collide_simd<const L: usize>(
        &self,
        base: usize,
        count: usize,
        center: &[A; K],
        rsq: A,
    ) -> bool
    where
        Simd<A, L>: AxisSimd<L>,
    {
        // TODO: make voxels be SIMD-aligned to make loading efficient and avoid a check for the
        // back elements of the voxel's collision buffer
        let center_simd: [Simd<A, L>; K] = array::from_fn(|k| Simd::splat(center[k]));
        let rsq_simd = Simd::splat(rsq);

        let mut i = 0;
        while i + L <= count {
            let mut distsq = Simd::splat(A::ZERO);
            for (k, &c) in center_simd.iter().enumerate() {
                let chunk = Simd::from_slice(&self.points[base + k * count + i..]);
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
                let diff = self.points[base + k * count + i] - c;
                distsq = distsq + diff.square();
            }
            distsq <= rsq
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let t = Mvt::<2>::new(&points, 0.2);
        println!("{t:?}");
    }

    #[test]
    fn exact_query_single() {
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let t = Mvt::<2>::new(&points, 0.2);

        let q0 = [0.0, -0.01];
        assert!(t.collides(&q0, 0.12));
    }

    #[test]
    fn no_collision() {
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let t = Mvt::<2>::new(&points, 0.2);

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

        let t = Mvt::<3>::new(&points, 0.2);

        assert!(t.collides(&[0.0, 0.1, 0.0], 0.11));
        assert!(!t.collides(&[0.0, 0.1, 0.0], 0.05));
    }

    #[test]
    fn point_radius() {
        let points = [[0.0, 0.0], [0.0, 1.0]];
        let r_max = 1.0;

        let mvt = Mvt::<2>::with_point_radius(&points, r_max, 0.5);
        assert!(mvt.collides(&[0.6, 0.0], 0.2));
        assert!(!mvt.collides(&[0.6, 0.0], 0.05));
    }

    #[test]
    fn custom_index_type() {
        const R: f32 = 0.04;
        let points = [[0.0, 0.1], [0.4, -0.2], [-0.2, -0.1]];
        let mut rng = SmallRng::seed_from_u64(1234);
        let t: Mvt<2, f32, u16> = Mvt::new(&points, R);

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

        let result = Mvt::<2, f32, u8>::try_new(&points, 0.1);
        assert_eq!(result.unwrap_err(), NewMvtError::TooManyVoxels);
    }

    #[test]
    fn non_cubic_workspace() {
        // a long, thin cloud: 100 units wide along x, but only 1 unit tall along y. A cubic grid
        // would need the same (huge) cell count along y as along x; a non-cubic grid can use far
        // fewer cells along y.
        #[expect(
            clippy::cast_precision_loss,
            reason = "loop index is tiny relative to f32's mantissa"
        )]
        let points: Vec<[f32; 2]> = (0..200_i32).map(|i| [i as f32 * 0.5, 0.3]).collect();
        let t = Mvt::<2>::new(&points, 0.05);

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
    fn empty_cloud() {
        let points: [[f32; 2]; 0] = [];
        let mvt = Mvt::<2>::new(&points, 1.0);
        assert!(!mvt.collides(&[0.0, 0.0], 100.0));
    }

    #[test]
    fn single_point() {
        let points = [[1.0, 1.0]];
        let mvt = Mvt::<2>::new(&points, 1.0);
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

        let mvt = Mvt::<2>::new(&points, 0.1);
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
        let t = Mvt::<2>::new(&points, R);

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
        let t = Mvt::<3>::with_point_radius(&points, R, 0.05);

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
        let t = Mvt::<2>::new(&points, R);

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
        let t = Mvt::<3>::with_point_radius(&points, R, R_POINT);

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
        let t = Mvt::<2>::new(&points, 0.2);

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
        let t = Mvt::<2>::new(&points, 0.2);

        let centers = [
            Simd::from_array([10.0, -10.0, 20.0, -20.0]),
            Simd::from_array([10.0, -10.0, 20.0, -20.0]),
        ];
        let radii = Simd::splat(0.1);

        assert!(!t.collides_simd(&centers, radii));
    }
}
