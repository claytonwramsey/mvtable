//! Renders an animated figure in Rerun showing how [`Mvt`](mvtable::Mvt)'s block-search collision
//! query examines the sparse voxel grid.

use std::{collections::HashMap, error::Error};

use mvtable::MutableMvt;
use mvtable_viz::{
    Aabb, K, Point, distsq, grid_params, point_cloud, point_to_coords, queries, r_max, v3,
};
use rerun::{
    Boxes3D, Color, Ellipsoids3D, Points3D, RecordingStream, RecordingStreamBuilder, TextLog,
    components::FillMode,
};

/// The world-space bounds of grid cell `coords`.
fn cell_bounds(coords: [usize; K], lo: Point, cell_size: Point) -> Aabb {
    let cell_lo: Point = std::array::from_fn(|k| lo[k] + coords[k] as f32 * cell_size[k]);
    Aabb {
        lo: cell_lo,
        hi: std::array::from_fn(|k| cell_lo[k] + cell_size[k]),
    }
}

/// Mirrors `mvtable`'s `search_block`: the range of grid coordinates whose cells could contain a
/// point within `radius` of `center`.
fn search_block(
    center: Point,
    radius: f32,
    workspace_lo: Point,
    scale: Point,
    grid_width: [usize; K],
) -> ([usize; K], [usize; K]) {
    let mut bmin = [0usize; K];
    let mut bmax = [0usize; K];
    for k in 0..K {
        let grid_max = grid_width[k] - 1;
        let rg = radius * scale[k];
        let v = (center[k] - workspace_lo[k]) * scale[k];
        bmin[k] = ((v - rg) as usize).min(grid_max);
        bmax[k] = ((v + rg) as usize).min(grid_max);
    }
    (bmin, bmax)
}

/// The next grid coordinate after `coords` in `mvtable`'s odometer search order (axis 0 fastest),
/// or `None` once the whole `[bmin, bmax]` block has been visited.
fn next_coords(mut coords: [usize; K], bmin: [usize; K], bmax: [usize; K]) -> Option<[usize; K]> {
    for dim in 0..K {
        coords[dim] += 1;
        if coords[dim] <= bmax[dim] {
            return Some(coords);
        }
        coords[dim] = bmin[dim];
    }
    None
}

#[expect(
    clippy::too_many_arguments,
    reason = "internal animation helper, not a public API"
)]
fn run_query(
    rec: &RecordingStream,
    mut step: i64,
    name: &str,
    center: Point,
    radius: f32,
    aabb: &Aabb,
    grid_width: [usize; K],
    scale: Point,
    cell_size: Point,
    voxel_points: &HashMap<[usize; K], Vec<Point>>,
    voxel_aabb: &HashMap<[usize; K], Aabb>,
) -> Result<(i64, bool), Box<dyn Error>> {
    let prefix = format!("query/{name}");
    let rsq = radius * radius;
    let log_text = |rec: &RecordingStream, step: i64, text: String| -> Result<(), Box<dyn Error>> {
        rec.set_time_sequence("query_step", step);
        rec.log(format!("{prefix}/log"), &TextLog::new(text))?;
        Ok(())
    };
    let log_sphere =
        |rec: &RecordingStream, step: i64, color: Color| -> Result<(), Box<dyn Error>> {
            rec.set_time_sequence("query_step", step);
            rec.log(
                format!("{prefix}/sphere"),
                &Ellipsoids3D::from_centers_and_radii([v3(center)], [radius]).with_colors([color]),
            )?;
            Ok(())
        };

    log_sphere(rec, step, Color::from_rgb(220, 220, 220))?;
    log_text(
        rec,
        step,
        format!("query '{name}': center={center:?}, radius={radius:.3}; testing global AABB cull"),
    )?;
    step += 1;

    if aabb.closest_distsq_to(&center) > rsq {
        log_text(
            rec,
            step,
            "culled: sphere doesn't reach the point cloud's bounding box".into(),
        )?;
        log_sphere(rec, step, Color::from_rgb(90, 200, 120))?;
        return Ok((step + 1, false));
    }
    log_text(
        rec,
        step,
        "cull passed; computing the block of voxels the sphere could reach".into(),
    )?;
    step += 1;

    let (bmin, bmax) = search_block(center, radius, aabb.lo, scale, grid_width);
    rec.set_time_sequence("query_step", step);
    rec.log(
        format!("{prefix}/search_block"),
        &Boxes3D::from_mins_and_sizes(
            [v3(cell_bounds(bmin, aabb.lo, cell_size).lo)],
            [v3(std::array::from_fn(|k| {
                (bmax[k] - bmin[k] + 1) as f32 * cell_size[k]
            }))],
        )
        .with_colors([Color::from_rgb(230, 160, 60)])
        .with_fill_mode(FillMode::MajorWireframe),
    )?;
    log_text(
        rec,
        step,
        format!(
            "search block: {bmin:?}..={bmax:?} ({} cells)",
            (0..K).map(|k| bmax[k] - bmin[k] + 1).product::<usize>()
        ),
    )?;
    step += 1;

    let mut visited_mins = Vec::new();
    let mut checked_points = Vec::new();
    let mut hit_point = None;
    let mut coords = bmin;
    'search: loop {
        rec.set_time_sequence("query_step", step);
        let cell = cell_bounds(coords, aabb.lo, cell_size);
        rec.log(
            format!("{prefix}/current_cell"),
            &Boxes3D::from_mins_and_sizes([v3(cell.lo)], [v3(cell_size)])
                .with_colors([Color::from_rgb(255, 220, 90)])
                .with_fill_mode(FillMode::MajorWireframe),
        )?;

        match voxel_points.get(&coords) {
            None => log_text(rec, step, format!("cell {coords:?}: empty, skip"))?,
            Some(pts) => {
                let vaabb = voxel_aabb[&coords];
                if vaabb.closest_distsq_to(&center) > rsq {
                    log_text(
                        rec,
                        step,
                        format!(
                            "cell {coords:?}: occupied, but its points' bounding box is out of range, skip"
                        ),
                    )?;
                } else {
                    log_text(
                        rec,
                        step,
                        format!(
                            "cell {coords:?}: occupied and in range, checking its {} point(s)",
                            pts.len()
                        ),
                    )?;
                    for p in pts {
                        checked_points.push(*p);
                        if distsq(p, &center) <= rsq {
                            hit_point = Some(*p);
                            break;
                        }
                    }
                }
            }
        }
        visited_mins.push(v3(cell.lo));
        rec.log(
            format!("{prefix}/visited_cells"),
            &Boxes3D::from_mins_and_sizes(visited_mins.clone(), [v3(cell_size)])
                .with_colors([Color::from_rgb(90, 110, 140)])
                .with_fill_mode(FillMode::MajorWireframe),
        )?;
        if !checked_points.is_empty() {
            rec.log(
                format!("{prefix}/checked_points"),
                &Points3D::new(checked_points.iter().copied().map(v3))
                    .with_colors([Color::from_rgb(200, 200, 90)])
                    .with_radii([cell_size.iter().copied().fold(f32::INFINITY, f32::min) * 0.08]),
            )?;
        }
        step += 1;

        if hit_point.is_some() {
            break 'search;
        }
        match next_coords(coords, bmin, bmax) {
            Some(c) => coords = c,
            None => break 'search,
        }
    }

    rec.set_time_sequence("query_step", step);
    rec.log(
        format!("{prefix}/current_cell"),
        &Boxes3D::from_mins_and_sizes(Vec::<(f32, f32, f32)>::new(), Vec::<(f32, f32, f32)>::new()),
    )?;
    let found = hit_point.is_some();
    if let Some(p) = hit_point {
        rec.log(
            format!("{prefix}/checked_points"),
            &Points3D::new(checked_points.iter().copied().map(v3))
                .with_colors(
                    checked_points
                        .iter()
                        .map(|&q| {
                            if q == p {
                                Color::from_rgb(230, 60, 60)
                            } else {
                                Color::from_rgb(200, 200, 90)
                            }
                        })
                        .collect::<Vec<_>>(),
                )
                .with_radii([cell_size.iter().copied().fold(f32::INFINITY, f32::min) * 0.08]),
        )?;
        log_text(
            rec,
            step,
            format!("collision: point {p:?} is within radius"),
        )?;
        log_sphere(rec, step, Color::from_rgb(220, 70, 70))?;
    } else {
        log_text(
            rec,
            step,
            "search block exhausted with no point in range: no collision".into(),
        )?;
        log_sphere(rec, step, Color::from_rgb(90, 200, 120))?;
    }
    step += 1;

    Ok((step, found))
}

fn main() -> Result<(), Box<dyn Error>> {
    let points = point_cloud();
    let aabb = Aabb::of(&points);
    let r_max = r_max(&aabb);
    let (grid_width, scale) = grid_params(&aabb, r_max);
    let cell_size: Point = std::array::from_fn(|k| 1.0 / scale[k]);

    let mut voxel_points: HashMap<[usize; K], Vec<Point>> = HashMap::new();
    for p in &points {
        voxel_points
            .entry(point_to_coords(p, aabb.lo, scale, grid_width))
            .or_default()
            .push(*p);
    }
    let voxel_aabb: HashMap<[usize; K], Aabb> = voxel_points
        .iter()
        .map(|(&c, pts)| (c, Aabb::of(pts)))
        .collect();

    // ground truth, from the real crate, to sanity-check the mirrored search logic below
    let mvt = MutableMvt::<K>::new(&points, r_max);

    let rec = RecordingStreamBuilder::new("mvtable_query").save("doc/mvt_query.rrd")?;

    rec.log_static(
        "workspace",
        &Boxes3D::from_mins_and_sizes([v3(aabb.lo)], [v3(aabb.size())])
            .with_colors([Color::from_rgb(60, 60, 60)])
            .with_fill_mode(FillMode::MajorWireframe),
    )?;
    rec.log_static(
        "points",
        &Points3D::new(points.iter().copied().map(v3))
            .with_colors([Color::from_rgb(120, 110, 80)])
            .with_radii([cell_size.iter().copied().fold(f32::INFINITY, f32::min) * 0.05]),
    )?;
    rec.log_static(
        "grid/cells",
        &Boxes3D::from_mins_and_sizes(
            voxel_points
                .keys()
                .map(|&c| v3(cell_bounds(c, aabb.lo, cell_size).lo))
                .collect::<Vec<_>>(),
            [v3(cell_size)],
        )
        .with_colors([Color::from_rgb(50, 70, 100)])
        .with_fill_mode(FillMode::MajorWireframe),
    )?;

    let mut step = 0i64;
    for (name, center, radius) in queries(r_max) {
        let expected = mvt.collides(&center, radius);
        let found;
        (step, found) = run_query(
            &rec,
            step,
            name,
            center,
            radius,
            &aabb,
            grid_width,
            scale,
            cell_size,
            &voxel_points,
            &voxel_aabb,
        )?;
        assert_eq!(
            found, expected,
            "mirrored search_block disagreed with the real Mvt::collides for query '{name}'"
        );
        println!("query '{name}': collides = {expected}");
        step += 20; // gap between queries so their timeline ranges don't run together
    }

    Ok(())
}
