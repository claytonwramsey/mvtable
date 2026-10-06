//! Renders an animated figure in Rerun showing a simplified CAPT (collision-affording point tree)
//! collision query.
use std::error::Error;

use mvtable_viz::{
    Aabb, Point, brute_force_collides, distsq, point_cloud, queries, r_max,
    tree::{self, Tree},
    v3,
};
use rerun::{
    Boxes3D, Color, Ellipsoids3D, Points3D, RecordingStream, RecordingStreamBuilder, TextLog,
    components::FillMode,
};

/// Leaf bucket size, matching [`kdtree_query`](../kdtree_query) so the two trees have the same
/// shape and only the traversal differs.
const LEAF_CAPACITY: usize = 8;

/// A leaf's precomputed affordance buffer.
struct Affordance {
    points: Vec<Point>,
    own_count: usize,
    bbox: Aabb,
}

fn build_affordance(tree: &Tree, leaf_idx: usize, all_points: &[Point], r_max: f32) -> Affordance {
    let node = &tree.nodes[leaf_idx];
    let rsq = r_max * r_max;
    let points: Vec<Point> = all_points
        .iter()
        .copied()
        .filter(|p| node.bbox.closest_distsq_to(p) <= rsq)
        .collect();
    let bbox = Aabb::of(&points);
    Affordance {
        own_count: node.points.len(),
        points,
        bbox,
    }
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
    tree: &Tree,
    all_points: &[Point],
    r_max: f32,
) -> Result<(i64, Option<Point>), Box<dyn Error>> {
    let prefix = format!("query/{name}");
    let rsq = radius * radius;
    let log_text = |rec: &RecordingStream, step: i64, text: String| -> Result<(), Box<dyn Error>> {
        rec.set_time_sequence("query_step", step);
        rec.log(format!("{prefix}/log"), &TextLog::new(text))?;
        Ok(())
    };

    rec.set_time_sequence("query_step", step);
    rec.log(
        format!("{prefix}/sphere"),
        &Ellipsoids3D::from_centers_and_radii([v3(center)], [radius])
            .with_colors([Color::from_rgb(220, 220, 220)]),
    )?;
    log_text(
        rec,
        step,
        format!(
            "query '{name}': center={center:?}, radius={radius:.3}; single descent, no bounding-box tests"
        ),
    )?;
    step += 1;

    let mut path_mins = Vec::new();
    let mut path_sizes = Vec::new();
    let mut idx = tree.root;
    let mut level = 0usize;
    loop {
        let node = &tree.nodes[idx];
        rec.set_time_sequence("query_step", step);
        rec.log(
            format!("{prefix}/current_node"),
            &Boxes3D::from_mins_and_sizes([v3(node.bbox.lo)], [v3(node.bbox.size())])
                .with_colors([Color::from_rgb(255, 220, 90)])
                .with_fill_mode(FillMode::MajorWireframe),
        )?;
        path_mins.push(v3(node.bbox.lo));
        path_sizes.push(v3(node.bbox.size()));
        rec.log(
            format!("{prefix}/path"),
            &Boxes3D::from_mins_and_sizes(path_mins.clone(), path_sizes.clone())
                .with_colors([Color::from_rgb(190, 140, 70)])
                .with_fill_mode(FillMode::MajorWireframe),
        )?;

        if node.is_leaf() {
            log_text(
                rec,
                step,
                format!(
                    "level {level}: reached leaf ({} own point(s))",
                    node.points.len()
                ),
            )?;
            step += 1;
            break;
        }
        let go_left = center[node.axis] <= node.test;
        log_text(
            rec,
            step,
            format!(
                "level {level}: compare center[axis {}]={:.3} to split {:.3} -> go {}",
                node.axis,
                center[node.axis],
                node.test,
                if go_left { "left" } else { "right" }
            ),
        )?;
        step += 1;
        idx = if go_left {
            node.left.unwrap()
        } else {
            node.right.unwrap()
        };
        level += 1;
    }

    let affordance = build_affordance(tree, idx, all_points, r_max);
    rec.set_time_sequence("query_step", step);
    rec.log(
        format!("{prefix}/affordance_bbox"),
        &Boxes3D::from_mins_and_sizes([v3(affordance.bbox.lo)], [v3(affordance.bbox.size())])
            .with_colors([Color::from_rgb(230, 160, 60)])
            .with_fill_mode(FillMode::MajorWireframe),
    )?;
    if affordance.bbox.closest_distsq_to(&center) > rsq {
        log_text(
            rec,
            step,
            "leaf's affordance bbox is out of range: no collision, no points scanned".into(),
        )?;
        rec.log(
            format!("{prefix}/sphere"),
            &Ellipsoids3D::from_centers_and_radii([v3(center)], [radius])
                .with_colors([Color::from_rgb(90, 200, 120)]),
        )?;
        return Ok((step + 1, None));
    }
    log_text(
        rec,
        step,
        format!(
            "leaf's affordance bbox in range: scanning its {} afforded point(s) ({} native to this leaf, {} borrowed from neighbors)",
            affordance.points.len(),
            affordance.own_count,
            affordance.points.len() - affordance.own_count,
        ),
    )?;
    step += 1;

    let mut checked = Vec::new();
    let mut hit = None;
    for p in &affordance.points {
        checked.push(*p);
        rec.set_time_sequence("query_step", step);
        rec.log(
            format!("{prefix}/checked_points"),
            &Points3D::new(checked.iter().copied().map(v3))
                .with_colors([Color::from_rgb(200, 200, 90)])
                .with_radii([0.03]),
        )?;
        step += 1;
        if distsq(p, &center) <= rsq {
            hit = Some(*p);
            break;
        }
    }

    rec.set_time_sequence("query_step", step);
    rec.log(
        format!("{prefix}/current_node"),
        &Boxes3D::from_mins_and_sizes(Vec::<(f32, f32, f32)>::new(), Vec::<(f32, f32, f32)>::new()),
    )?;
    if let Some(p) = hit {
        log_text(
            rec,
            step,
            format!("collision: point {p:?} is within radius"),
        )?;
        rec.log(
            format!("{prefix}/sphere"),
            &Ellipsoids3D::from_centers_and_radii([v3(center)], [radius])
                .with_colors([Color::from_rgb(220, 70, 70)]),
        )?;
    } else {
        log_text(
            rec,
            step,
            "affordance buffer exhausted with no point in range: no collision".into(),
        )?;
        rec.log(
            format!("{prefix}/sphere"),
            &Ellipsoids3D::from_centers_and_radii([v3(center)], [radius])
                .with_colors([Color::from_rgb(90, 200, 120)]),
        )?;
    }
    step += 1;

    Ok((step, hit))
}

fn main() -> Result<(), Box<dyn Error>> {
    let points = point_cloud();
    let aabb = Aabb::of(&points);
    let r_max = r_max(&aabb);
    let tree = tree::build(&points, LEAF_CAPACITY);

    let rec = RecordingStreamBuilder::new("capt_query").save("doc/capt_query.rrd")?;

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
            .with_radii([0.02]),
    )?;
    let leaf_mins: Vec<_> = tree
        .nodes
        .iter()
        .filter(|n| n.is_leaf())
        .map(|n| v3(n.bbox.lo))
        .collect();
    let leaf_sizes: Vec<_> = tree
        .nodes
        .iter()
        .filter(|n| n.is_leaf())
        .map(|n| v3(n.bbox.size()))
        .collect();
    rec.log_static(
        "tree/leaves",
        &Boxes3D::from_mins_and_sizes(leaf_mins, leaf_sizes)
            .with_colors([Color::from_rgb(80, 60, 100)])
            .with_fill_mode(FillMode::MajorWireframe),
    )?;

    let leaf_count = tree.nodes.iter().filter(|n| n.is_leaf()).count();
    let avg_affordance: f64 = tree
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.is_leaf())
        .map(|(i, _)| build_affordance(&tree, i, &points, r_max).points.len())
        .sum::<usize>() as f64
        / leaf_count as f64;
    let n_points = points.len();
    println!(
        "CAPT-like tree: {n_points} points, {leaf_count} leaves (capacity {LEAF_CAPACITY}), avg affordance buffer size {avg_affordance:.1}"
    );

    let mut step = 0i64;
    for (name, center, radius) in queries(r_max) {
        let expected = brute_force_collides(&points, center, radius);
        let hit;
        (step, hit) = run_query(&rec, step, name, center, radius, &tree, &points, r_max)?;
        assert_eq!(
            hit.is_some(),
            expected,
            "CAPT-like search disagreed with a brute-force scan for query '{name}'"
        );
        println!("query '{name}': collides = {expected}");
        step += 20;
    }

    Ok(())
}
