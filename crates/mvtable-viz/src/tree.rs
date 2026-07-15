//! A minimal balanced binary point tree, shared by the `kdtree_query` and `capt_query` figures.
//!
//! Both figures build the exact same tree shape (recursive median split, round-robin axis) since
//! that's the part a k-d tree and a CAPT have in common; what differs between the two figures is
//! how the tree is *searched* (see each binary's own traversal code), which is the point of
//! showing them side by side.

use crate::{Aabb, K, Point};

/// One node of a [`Tree`]. Internal nodes have `left`/`right` and no `points`; leaves have
/// `points` and no children.
pub struct Node {
    /// The bounding box of this node's own points (for a leaf) or the union of its children's
    /// boxes (for an internal node).
    pub bbox: Aabb,
    /// The axis this node was split on, if it's an internal node.
    pub axis: usize,
    /// The split value used to route a query into `left` (`<= test`) or `right` (`> test`), if
    /// this is an internal node.
    pub test: f32,
    pub left: Option<usize>,
    pub right: Option<usize>,
    /// This leaf's own points. Empty for internal nodes.
    pub points: Vec<Point>,
}

impl Node {
    #[must_use]
    pub const fn is_leaf(&self) -> bool {
        self.left.is_none()
    }
}

/// An arena-allocated balanced binary tree over a point cloud.
pub struct Tree {
    pub nodes: Vec<Node>,
    pub root: usize,
}

/// Build a [`Tree`] over `points` by recursively splitting on the median of a round-robin axis,
/// stopping once a node holds `leaf_capacity` points or fewer.
#[must_use]
pub fn build(points: &[Point], leaf_capacity: usize) -> Tree {
    let mut nodes = Vec::new();
    let mut owned: Vec<Point> = points.to_vec();
    let root = build_node(&mut nodes, &mut owned, 0, leaf_capacity);
    Tree { nodes, root }
}

fn build_node(
    nodes: &mut Vec<Node>,
    points: &mut [Point],
    axis: usize,
    leaf_capacity: usize,
) -> usize {
    let bbox = Aabb::of(points);
    if points.len() <= leaf_capacity {
        let idx = nodes.len();
        nodes.push(Node {
            bbox,
            axis,
            test: 0.0,
            left: None,
            right: None,
            points: points.to_vec(),
        });
        return idx;
    }

    let mid = points.len() / 2;
    points.select_nth_unstable_by(mid, |a, b| a[axis].total_cmp(&b[axis]));
    let test = points[mid][axis];
    let next_axis = (axis + 1) % K;

    let (lhs, rhs) = points.split_at_mut(mid);
    let left = build_node(nodes, lhs, next_axis, leaf_capacity);
    let right = build_node(nodes, rhs, next_axis, leaf_capacity);
    let bbox = Aabb {
        lo: std::array::from_fn(|k| nodes[left].bbox.lo[k].min(nodes[right].bbox.lo[k])),
        hi: std::array::from_fn(|k| nodes[left].bbox.hi[k].max(nodes[right].bbox.hi[k])),
    };

    let idx = nodes.len();
    nodes.push(Node {
        bbox,
        axis,
        test,
        left: Some(left),
        right: Some(right),
        points: Vec::new(),
    });
    idx
}
