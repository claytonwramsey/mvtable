//! Renders an animated figure in Rerun showing a classic k-d tree's ball-search collision query.

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

/// Leaf/node bucket size, chosen to land in the same ballpark of cell count as the MVT figure's
/// voxel grid, so the two are visually comparable.
const LEAF_CAPACITY: usize = 8;

struct Search<'a> {
    rec: &'a RecordingStream,
    prefix: String,
    step: i64,
    visited_mins: Vec<(f32, f32, f32)>,
    visited_sizes: Vec<(f32, f32, f32)>,
    visited_colors: Vec<Color>,
    checked_points: Vec<Point>,
}

impl Search<'_> {
    fn log_text(&self, text: String) -> Result<(), Box<dyn Error>> {
        self.rec.set_time_sequence("query_step", self.step);
        self.rec
            .log(format!("{}/log", self.prefix), &TextLog::new(text))?;
        Ok(())
    }

    fn log_current(&self, bbox: &Aabb) -> Result<(), Box<dyn Error>> {
        self.rec.set_time_sequence("query_step", self.step);
        self.rec.log(
            format!("{}/current_node", self.prefix),
            &Boxes3D::from_mins_and_sizes([v3(bbox.lo)], [v3(bbox.size())])
                .with_colors([Color::from_rgb(255, 220, 90)])
                .with_fill_mode(FillMode::MajorWireframe),
        )?;
        Ok(())
    }

    fn log_visited(&mut self, bbox: &Aabb, color: Color) -> Result<(), Box<dyn Error>> {
        self.visited_mins.push(v3(bbox.lo));
        self.visited_sizes.push(v3(bbox.size()));
        self.visited_colors.push(color);
        self.rec.set_time_sequence("query_step", self.step);
        self.rec.log(
            format!("{}/visited_nodes", self.prefix),
            &Boxes3D::from_mins_and_sizes(self.visited_mins.clone(), self.visited_sizes.clone())
                .with_colors(self.visited_colors.clone())
                .with_fill_mode(FillMode::MajorWireframe),
        )?;
        Ok(())
    }

    fn log_checked_points(&self) -> Result<(), Box<dyn Error>> {
        self.rec.set_time_sequence("query_step", self.step);
        self.rec.log(
            format!("{}/checked_points", self.prefix),
            &Points3D::new(self.checked_points.iter().copied().map(v3))
                .with_colors([Color::from_rgb(200, 200, 90)])
                .with_radii([0.03]),
        )?;
        Ok(())
    }

    /// Recursively search `tree` starting at `node_idx`.
    fn search(
        &mut self,
        tree: &Tree,
        node_idx: usize,
        center: Point,
        rsq: f32,
    ) -> Result<Option<Point>, Box<dyn Error>> {
        let node = &tree.nodes[node_idx];
        self.log_current(&node.bbox)?;

        if node.bbox.closest_distsq_to(&center) > rsq {
            self.log_text(format!(
                "node bbox {:?}..{:?}: too far, prune subtree",
                node.bbox.lo, node.bbox.hi
            ))?;
            self.log_visited(&node.bbox, Color::from_rgb(90, 90, 90))?;
            self.step += 1;
            return Ok(None);
        }

        if node.is_leaf() {
            self.log_text(format!(
                "leaf with {} point(s): in range, checking each",
                node.points.len()
            ))?;
            self.log_visited(&node.bbox, Color::from_rgb(190, 110, 200))?;
            self.step += 1;
            for p in &node.points {
                self.checked_points.push(*p);
                self.step += 1;
                self.log_checked_points()?;
                if distsq(p, &center) <= rsq {
                    return Ok(Some(*p));
                }
            }
            return Ok(None);
        }

        self.log_text(format!(
            "internal node bbox {:?}..{:?}: in range, descend",
            node.bbox.lo, node.bbox.hi
        ))?;
        self.log_visited(&node.bbox, Color::from_rgb(120, 90, 170))?;
        self.step += 1;

        let (near, far) = if center[node.axis] <= node.test {
            (node.left.unwrap(), node.right.unwrap())
        } else {
            (node.right.unwrap(), node.left.unwrap())
        };
        if let Some(hit) = self.search(tree, near, center, rsq)? {
            return Ok(Some(hit));
        }
        self.search(tree, far, center, rsq)
    }
}

fn run_query(
    rec: &RecordingStream,
    mut step: i64,
    name: &str,
    center: Point,
    radius: f32,
    tree: &Tree,
) -> Result<(i64, Option<Point>), Box<dyn Error>> {
    let prefix = format!("query/{name}");
    rec.set_time_sequence("query_step", step);
    rec.log(
        format!("{prefix}/sphere"),
        &Ellipsoids3D::from_centers_and_radii([v3(center)], [radius])
            .with_colors([Color::from_rgb(220, 220, 220)]),
    )?;
    rec.log(
        format!("{prefix}/log"),
        &TextLog::new(format!(
            "query '{name}': center={center:?}, radius={radius:.3}; descending from the root"
        )),
    )?;
    step += 1;

    let mut search = Search {
        rec,
        prefix: prefix.clone(),
        step,
        visited_mins: Vec::new(),
        visited_sizes: Vec::new(),
        visited_colors: Vec::new(),
        checked_points: Vec::new(),
    };
    let hit = search.search(tree, tree.root, center, radius * radius)?;
    step = search.step;

    rec.set_time_sequence("query_step", step);
    rec.log(
        format!("{prefix}/current_node"),
        &Boxes3D::from_mins_and_sizes(Vec::<(f32, f32, f32)>::new(), Vec::<(f32, f32, f32)>::new()),
    )?;
    if let Some(p) = hit {
        rec.log(
            format!("{prefix}/log"),
            &TextLog::new(format!("collision: point {p:?} is within radius")),
        )?;
        rec.log(
            format!("{prefix}/sphere"),
            &Ellipsoids3D::from_centers_and_radii([v3(center)], [radius])
                .with_colors([Color::from_rgb(220, 70, 70)]),
        )?;
    } else {
        rec.log(
            format!("{prefix}/log"),
            &TextLog::new("search exhausted with no point in range: no collision"),
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

    let rec = RecordingStreamBuilder::new("kdtree_query").save("doc/kdtree_query.rrd")?;

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

    println!(
        "k-d tree: {} points, {} nodes ({} leaves, leaf capacity {LEAF_CAPACITY})",
        points.len(),
        tree.nodes.len(),
        tree.nodes.iter().filter(|n| n.is_leaf()).count()
    );

    let mut step = 0i64;
    for (name, center, radius) in queries(r_max) {
        let expected = brute_force_collides(&points, center, radius);
        let hit;
        (step, hit) = run_query(&rec, step, name, center, radius, &tree)?;
        assert_eq!(
            hit.is_some(),
            expected,
            "k-d tree search disagreed with a brute-force scan for query '{name}'"
        );
        println!("query '{name}': collides = {expected}");
        step += 20;
    }

    Ok(())
}
