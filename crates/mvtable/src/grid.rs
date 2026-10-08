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

/// Given a bounding box over a point cloud and a voxel width `cell_wd`, compute the per-axis grid
/// width (in both `usize` and `I` form) and the coordinate scale factor used to map a point into
/// grid indices.
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
///
/// If this returns `Err`, `tables` is left exactly as it was before the call.
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
            let new_offset_i = I::from_usize(new_offset).ok_or(TooManyVoxels)?;
            tables.resize(new_offset + grid_width[level + 1], I::SENTINEL);
            tables[slot] = new_offset_i;
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

/// Return an upper bound on the number of entries [`get_leaf`] can append to `tables` while
/// assigning `n_points` points to a grid of width `grid_width`.
///
/// Each point creates at most one new subtable per level.
/// Each slot in a parent table holds at most one subtable.
/// The bound takes the smaller of these two limits at every level, so it stays small for sparse
/// clouds and approaches the size of the fully dense grid only for dense ones.
#[must_use]
pub fn subtable_capacity_bound<const K: usize>(grid_width: [usize; K], n_points: usize) -> usize {
    let mut total = 0usize;
    let mut reachable = grid_width[0];
    for &width in &grid_width[1..] {
        reachable = reachable.min(n_points).saturating_mul(width);
        total = total.saturating_add(reachable);
    }
    total
}

/// Return an upper bound on the number of voxels that `n_points` points can occupy in a grid of
/// width `grid_width`.
#[must_use]
pub fn max_voxels<const K: usize>(grid_width: [usize; K], n_points: usize) -> usize {
    grid_width
        .iter()
        .try_fold(1usize, |acc, &w| acc.checked_mul(w))
        .map_or(n_points, |cells| cells.min(n_points))
}

/// The largest number of grid cells per point for which [`assign_points`] uses a flat array during
/// construction.
const FLAT_ARRAY_CELLS_PER_POINT: usize = 8;

/// Map grid coordinates `coords` to a unique index in `0..grid_width.iter().product()`.
fn linearize<const K: usize>(coords: [usize; K], grid_width: [usize; K]) -> usize {
    coords
        .iter()
        .zip(&grid_width)
        .fold(0, |acc, (&c, &w)| acc * w + c)
}

/// Find the leaf-level table slot for every point in `points`, in order, and call `assign` on
/// each point together with its slot.
///
/// `assign` must leave the slot holding the index of the voxel the point belongs to.
/// A slot holding [`Index::SENTINEL`] means the point's voxel does not exist yet.
///
/// `tables` must already contain at least `grid_width[0]` entries, as for [`get_leaf`].
/// On success, `tables` holds the full hierarchy for every point in `points`.
pub fn assign_points<A: Axis, I: Index, E: From<TooManyVoxels>, const K: usize>(
    tables: &mut Vec<I>,
    points: &[[A; K]],
    lo: [A; K],
    scale: [A; K],
    grid_width: [usize; K],
    mut assign: impl FnMut(&[A; K], &mut I) -> Result<(), E>,
) -> Result<(), E> {
    let n_cells = grid_width
        .iter()
        .try_fold(1usize, |acc, &w| acc.checked_mul(w))
        .filter(|&cells| cells <= FLAT_ARRAY_CELLS_PER_POINT.saturating_mul(points.len()))
        // the flat array only knows about voxels created here, so it needs an empty hierarchy
        .filter(|_| tables.len() == grid_width[0] && tables.iter().all(|&t| t == I::SENTINEL));

    let Some(n_cells) = n_cells else {
        // Reserving the worst case once is cheaper than growing `tables` one subtable at a time.
        tables.reserve(subtable_capacity_bound(grid_width, points.len()));
        for p in points {
            let leaf_slot = get_leaf(
                tables,
                grid_width,
                point_to_grid_coords(p, lo, scale, grid_width),
            )?;
            assign(p, &mut tables[leaf_slot])?;
        }
        return Ok(());
    };

    // The grid is small enough to use a flat array during construction, with one slot per cell.
    // Each point costs a single lookup rather than a walk down the sparse hierarchy.
    // The hierarchy is then built once per voxel in first-encounter order, so subtables are
    // allocated in first-encounter order.
    let mut cell_slots = vec![I::SENTINEL; n_cells];
    let mut new_cells: Vec<[usize; K]> = Vec::with_capacity(max_voxels(grid_width, points.len()));
    for p in points {
        let coords = point_to_grid_coords(p, lo, scale, grid_width);
        let slot = &mut cell_slots[linearize(coords, grid_width)];
        if *slot == I::SENTINEL {
            new_cells.push(coords);
        }
        assign(p, slot)?;
    }

    tables.reserve(subtable_capacity_bound(grid_width, new_cells.len()));
    for coords in new_cells {
        let leaf_slot = get_leaf(tables, grid_width, coords)?;
        tables[leaf_slot] = cell_slots[linearize(coords, grid_width)];
    }
    Ok(())
}
