#![cfg_attr(not(feature = "std"), no_std)]
#![warn(clippy::pedantic, clippy::nursery)]
#![warn(clippy::allow_attributes, reason = "prefer expect over allow")]
#![cfg_attr(doc, feature(rustdoc_missing_doc_code_examples))]
#![warn(missing_docs, rustdoc::missing_doc_code_examples)]
#![feature(portable_simd)]

use std::marker::PhantomData;

/// A multilevel voxel tree, a structure for point cloud collision checking.
///
/// The MVT can be used for fast, SIMD-parallel collision checking between spheres and point cloud
/// data.
///
/// # Generic parameters
///
/// - `T`: The underlying float type to use (typically, either `f32` or `f64`).
/// - `K`: The dimension of the space.
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
pub struct Mvt<T, const K: usize> {
    /* TODO: populate this structure */
    _phantom: PhantomData<[T; K]>, // phantom for no compiler errors
}
