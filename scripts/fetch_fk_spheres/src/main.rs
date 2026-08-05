//! Dumps the Fetch's ground-truth FK'd collision spheres (via `carom`'s generated fkcc, the
//! same code mvtable-bench's `true_max_query_radius` etc. were traced from) as a JSON array of
//! `{"x", "y", "z", "r"}` objects on stdout, for `fetch_bvh_figure.py` to render.
#![feature(portable_simd)]

use std::simd::Simd;

use carom::{Block, Robot, robot::Fetch};

/// The "arm_with_torso" goal pose from resources/fetch/problems/bookshelf_small_fetch/
/// request0001.yaml, in `Fetch::JOINT_NAMES` order — kept in sync by hand with
/// `fetch_bvh_figure.py`'s `READY_CFG`.
const READY_CFG: [f32; Fetch::DIM] = [
    0.05580749394926036,
    0.2319594187719277,
    -0.7632272745271215,
    0.7273950815863892,
    1.421938462868271,
    2.57193125373631,
    0.4549567580598895,
    -3.141592599877235,
];

fn main() {
    let block = Block(READY_CFG.map(Simd::<f32, 1>::splat));
    let mut spheres = Vec::new();
    Fetch.sphere_fk(&block, &mut spheres);

    println!("[");
    let n = spheres.len();
    for (i, sphere) in spheres.iter().enumerate() {
        let [x, y, z] = sphere.pos.map(|v| v[0]);
        let r = sphere.r[0];
        let comma = if i + 1 < n { "," } else { "" };
        println!("  {{\"x\": {x}, \"y\": {y}, \"z\": {z}, \"r\": {r}}}{comma}");
    }
    println!("]");
}
