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
//! construction, they take in a list of points in a point cloud and an voxel width used
//! to size the grid's voxels.
//!
//! ```rust
//! use mvtable::Mvt;
//!
//! // list of points in cloud
//! let points = [[0.0, 1.1], [0.2, 3.1]];
//! let voxel_width = 2.0;
//!
//! let mvt = Mvt::<2>::new(&points, voxel_width);
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

use core::ops::{Add, Div, Mul, Sub};

mod grid;
mod immutable;
mod mutable;

pub use immutable::{Mvt, NewMvtError};
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

/// Block width (in points) that each voxel's per-axis point buffer is padded to a multiple of
/// during [`Mvt`](crate::Mvt) construction, and that the scalar scan in [`scan_block`] processes
/// at a time.
const SCAN_BLOCK: usize = 8;

/// Determine whether any of the points described by `axes` (one contiguous per-axis slice each,
/// all the same length) lies within a squared distance of `rsq` from `center`
fn scan_block<A: Axis, const K: usize, const BLOCK: usize>(
    axes: &[&[A]; K],
    center: &[A; K],
    rsq: A,
) -> bool {
    let count = axes[0].len();

    let mut i = 0;
    // Autovectorizable loop: iterate over blocks, then calculate distances per block
    while i + BLOCK <= count {
        let mut distsq = [A::ZERO; BLOCK];
        for (k, &c) in center.iter().enumerate() {
            for (d, &p) in distsq.iter_mut().zip(&axes[k][i..i + BLOCK]) {
                let diff = p - c;
                *d = *d + diff.square();
            }
        }
        if distsq.iter().any(|&d| d <= rsq) {
            return true;
        }
        i += BLOCK;
    }

    // fewer than `BLOCK` points remain: fall back to a one-at-a-time scalar check for the
    // remainder. When `count` is itself a multiple of `BLOCK` (e.g. every `Mvt` voxel, which is
    // always padded to one), this range is empty and never runs.
    (i..count).any(|i| {
        let mut distsq = A::ZERO;
        for (k, &c) in center.iter().enumerate() {
            let diff = axes[k][i] - c;
            distsq = distsq + diff.square();
        }
        distsq <= rsq
    })
}


