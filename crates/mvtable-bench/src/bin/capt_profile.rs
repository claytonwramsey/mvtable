//! Focused profiling harness: load one workload, build a `capt` tree, and replay its scalar
//! query trace many times in a tight loop so `perf record` attributes self-time inside `capt`.
//!
//! Usage: `capt_profile <points.bin> <queries.bin> <filter_scale> [iters] [simd]`
//! Pass a 5th arg `simd` to profile `collides_simd` (L=8 batches) instead of scalar `collides`.
#![feature(portable_simd)]

use std::{
    env,
    fs::File,
    hint::black_box,
    io::{BufReader, Read},
    path::Path,
    simd::Simd,
};

use mvtable_bench::{SimdStructure, Structure, filter};

fn read_points(path: impl AsRef<Path>) -> Vec<[f32; 3]> {
    let mut r = BufReader::new(File::open(path).unwrap());
    let mut cb = [0u8; 8];
    r.read_exact(&mut cb).unwrap();
    let count = u64::from_le_bytes(cb) as usize;
    let mut pts = Vec::with_capacity(count);
    let mut buf = [0u8; 12];
    for _ in 0..count {
        r.read_exact(&mut buf).unwrap();
        pts.push([
            f32::from_le_bytes(buf[0..4].try_into().unwrap()),
            f32::from_le_bytes(buf[4..8].try_into().unwrap()),
            f32::from_le_bytes(buf[8..12].try_into().unwrap()),
        ]);
    }
    pts
}

fn read_queries(path: impl AsRef<Path>) -> Vec<([f32; 3], f32)> {
    let mut r = BufReader::new(File::open(path).unwrap());
    let mut cb = [0u8; 8];
    r.read_exact(&mut cb).unwrap();
    let count = u64::from_le_bytes(cb) as usize;
    let mut q = Vec::with_capacity(count);
    let mut buf = [0u8; 18];
    for _ in 0..count {
        r.read_exact(&mut buf).unwrap();
        q.push((
            [
                f32::from_le_bytes(buf[0..4].try_into().unwrap()),
                f32::from_le_bytes(buf[4..8].try_into().unwrap()),
                f32::from_le_bytes(buf[8..12].try_into().unwrap()),
            ],
            f32::from_le_bytes(buf[12..16].try_into().unwrap()),
        ));
    }
    q
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let points_path = &args[1];
    let queries_path = &args[2];
    let scale: f32 = args[3].parse().unwrap();
    let iters: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2000);

    let full = read_points(points_path);
    let queries = read_queries(queries_path);
    let r_max = queries.iter().fold(0.0f32, |m, q| m.max(q.1));
    let r_min = queries.iter().fold(f32::INFINITY, |m, q| m.min(q.1));

    let points = filter::centervox_filter(&full, scale * r_min);
    eprintln!(
        "points {} -> filtered {}, {} queries, r=({r_min},{r_max}), iters {iters}",
        full.len(),
        points.len(),
        queries.len()
    );

    let capt = capt::Capt::<3, f32, u32>::new(&points, (r_min, r_max), 8);

    let simd_mode = args.get(5).map(|s| s == "simd").unwrap_or(false);

    let mut hits = 0u64;
    if simd_mode {
        const L: usize = 8;
        let batches: Vec<([Simd<f32, L>; 3], Simd<f32, L>)> = queries
            .chunks_exact(L)
            .map(|ch| {
                let centers: [Simd<f32, L>; 3] =
                    std::array::from_fn(|k| Simd::from_array(std::array::from_fn(|l| ch[l].0[k])));
                let radii = Simd::from_array(std::array::from_fn(|l| ch[l].1));
                (centers, radii)
            })
            .collect();
        for _ in 0..iters {
            for (c, r) in &batches {
                if black_box(SimdStructure::collides_simd(&capt, c, *r)) {
                    hits += 1;
                }
            }
        }
    } else {
        for _ in 0..iters {
            for (c, r) in &queries {
                if black_box(Structure::collides(&capt, c, *r)) {
                    hits += 1;
                }
            }
        }
    }
    println!("hits {hits}");
}
