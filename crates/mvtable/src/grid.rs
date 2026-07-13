//! Grid-indexing helpers shared by [`crate::Mvt`] and [`crate::MutableMvt`]. Both structures use
//! the same table hierarchy and the same point-to-voxel-coordinate
//! mapping; this module is the one place that logic lives.

use alloc::{vec, vec::Vec};
use core::array;

use crate::{Aabb, Axis, Index};

/// Marker for "ran out of index space building the grid", returned by this module's helpers.
pub struct TooManyVoxels;

/// The result of [`size_grid`]: the per-axis grid width (in `usize` and `I` form) and the
/// coordinate scale factor used to map a point into grid indices.
pub type GridSizing<A, I, const K: usize> = ([usize; K], [I; K], [A; K]);

/// Given a bounding box over a point cloud and the combined query-plus-point radius `cell_wd`,
/// compute the per-axis grid width (in both `usize` and `I` form) and the coordinate scale factor
/// used to map a point into grid indices.
pub fn size_grid<A: Axis, I: Index, const K: usize>(
    aabb: &Aabb<A, K>,
    cell_wd: A,
) -> Result<GridSizing<A, I, K>, TooManyVoxels> {
    // size each axis independently
    let mut grid_width = [0usize; K];
    let mut grid_width_i = [I::ZERO; K];
    let mut scale = [A::ZERO; K];
    for k in 0..K {
        let extent = aabb.hi[k] - aabb.lo[k];
        // an extent of zero (e.g. every point shares this coordinate) would otherwise divide by
        // zero below, so round up to 1
        let extent = if extent > A::ZERO { extent } else { cell_wd };

        let gw = usize::max(1, (extent / cell_wd).to_index());
        grid_width[k] = gw;
        grid_width_i[k] = I::from_usize(gw).ok_or(TooManyVoxels)?;
        scale[k] = A::from_usize(gw) / extent;
    }
    Ok((grid_width, grid_width_i, scale))
}

/// Map point `p` into grid coordinates, given the grid's origin `lo`, `scale`, and `grid_width`.
///
/// A coordinate that would fall outside `0..grid_width[k]` (because `p` lies outside the box the
/// grid was originally sized for) is clamped to the nearest edge voxel along that axis, rather
/// than panicking.
/// The caller is still responsible for storing `p`'s true coordinates rather than
/// this clamped bucket, so query correctness is unaffected.
pub fn point_to_grid_coords<A: Axis, const K: usize>(
    p: &[A; K],
    lo: [A; K],
    scale: [A; K],
    grid_width: [usize; K],
) -> [usize; K] {
    array::from_fn(|k| {
        let v = (p[k] - lo[k]) * scale[k];
        v.to_index().min(grid_width[k] - 1)
    })
}

/// Descend the sparse table hierarchy for grid coordinates `coords`, allocating new subtables
/// (filled with [`Index::SENTINEL`]) as needed, and return the offset of the leaf-level table
/// slot that indexes into voxel storage.
///
/// `tables` must already contain at least `grid_width[0]` entries (the root table) before the
/// first call.
pub fn get_leaf<I: Index, const K: usize>(
    tables: &mut Vec<I>,
    grid_width: [usize; K],
    coords: [usize; K],
) -> Result<usize, TooManyVoxels> {
    let mut table_offset = 0usize;
    for (level, &coord) in coords[..K - 1].iter().enumerate() {
        let slot = table_offset + coord;
        if tables[slot] == I::SENTINEL {
            let new_offset = tables.len();
            tables.resize(new_offset + grid_width[level + 1], I::SENTINEL);
            tables[slot] = I::from_usize(new_offset).ok_or(TooManyVoxels)?;
        }
        table_offset = tables[slot].to_usize();
    }
    Ok(table_offset + coords[K - 1])
}

/// Build a fresh root table, sized to `grid_width[0]` entries and filled with
/// [`Index::SENTINEL`].
pub fn new_root_table<I: Index, const K: usize>(grid_width: [usize; K]) -> Vec<I> {
    vec![I::SENTINEL; grid_width[0]]
}
