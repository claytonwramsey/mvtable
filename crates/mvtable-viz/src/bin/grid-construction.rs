//! Renders an animated figure in Rerun showing an [`MutableMvt`]'s sparse voxel grid forming
//! around a synthetic point cloud.

use std::{array, collections::HashSet, error::Error};

use mvtable::MutableMvt;
use mvtable_viz::{Aabb, K, grid_params, point_cloud, point_to_coords, r_max, v3};
use rerun::{Boxes3D, Color, Points3D, RecordingStreamBuilder, components::FillMode};

fn main() -> Result<(), Box<dyn Error>> {
    let points = point_cloud();
    let aabb = Aabb::of(&points);
    let r_max = r_max(&aabb);

    let (grid_width, scale) = grid_params(&aabb, r_max);
    let cell_size: [f32; K] = array::from_fn(|k| 1.0 / scale[k]);

    let mut mvt = MutableMvt::<K>::with_workspace(aabb.lo, aabb.hi, r_max, 0.0);

    let rec = RecordingStreamBuilder::new("mvtable_grid_construction")
        .save("doc/mvt_grid_construction.rrd")?;

    rec.log_static(
        "workspace",
        &Boxes3D::from_mins_and_sizes([v3(aabb.lo)], [v3(aabb.size())])
            .with_colors([Color::from_rgb(90, 90, 90)])
            .with_fill_mode(FillMode::MajorWireframe),
    )?;

    let mut occupied = HashSet::new();
    let mut cell_mins = Vec::new();
    let mut logged_points = Vec::new();

    for (step, p) in points.iter().enumerate() {
        mvt.insert(p)?;
        logged_points.push(*p);

        let coords = point_to_coords(p, aabb.lo, scale, grid_width);
        if occupied.insert(coords) {
            cell_mins.push(v3(array::from_fn(|k| {
                aabb.lo[k] + coords[k] as f32 * cell_size[k]
            })));
        }

        rec.set_time_sequence("insertion_step", step as i64);
        rec.log(
            "points",
            &Points3D::new(logged_points.iter().copied().map(v3))
                .with_colors([Color::from_rgb(240, 220, 130)])
                .with_radii([cell_size.iter().copied().fold(f32::INFINITY, f32::min) * 0.05]),
        )?;
        rec.log(
            "grid/cells",
            &Boxes3D::from_mins_and_sizes(cell_mins.iter().copied(), [v3(cell_size)])
                .with_colors([Color::from_rgb(70, 140, 220)])
                .with_fill_mode(FillMode::MajorWireframe),
        )?;
    }

    assert_eq!(
        mvt.points().count(),
        points.len(),
        "every inserted point should be retrievable"
    );

    println!(
        "wrote doc/mvt_grid_construction.rrd: {} points into {} occupied voxels ({:?} grid)",
        points.len(),
        occupied.len(),
        grid_width,
    );
    Ok(())
}
