# MVT: Multilevel Voxel Tables

[![Crates.io page](https://img.shields.io/crates/v/mvtable)](https://crates.io/crates/mvtable)
[![Rust CI status](https://github.com/claytonwramsey/mvtable/actions/workflows/rust.yml/badge.svg)](https://github.com/claytonwramsey/mvtable/actions/workflows/rust.yml)

This is a Rust implementation of the _multilevel voxel table_ (MVT), a data structure
for fast collision checking between spheres and
point clouds.

If you use this in an academic work, please cite it as follows:

```bibtex
@inproceedings{chen2026vcc,
 author    = {Ching Chen and Tsung-Tai Yeh},
 title     = {VCC: Efficient Voxel-Based Collision Checking Framework for Real-Time Robotic
              Motion Planning},
 booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
 year      = {2026},
}
```

For further details, you can read my [blog post](https://claytonwramsey.com/blog/mvt) on this data structure.
Also, a paper explaining the internals of the data structure is available at the
[ICRA conference proceedings](https://rasevents.org/uploads/documents/pdfviewer/b1/d6/223112-5124.pdf).

## Usage

The core data structure in this library is the `Mvt`, a sparse voxel grid used for
collision checking. `Mvt`s are polymorphic over dimension and floating-point type. On
construction, they take in a list of points in a point cloud and a voxel width used
to size the grid's voxels.

```rust
use mvtable::Mvt;

// list of points in cloud
let points = [[0.0, 1.1], [0.2, 3.1]];
let voxel_width = 2.0;

let mvt = Mvt::<2>::new(&points, voxel_width);
```

Once you have an `Mvt`, you can use it for collision-checking against spheres.

```rust
use mvtable::Mvt;
let points = [[0.0, 1.1], [0.2, 3.1]];
let mvt = Mvt::<2>::new(&points, 2.0);
let center = [0.0, 0.0]; // center of sphere
let radius0 = 1.0; // radius of sphere
assert!(!mvt.collides(&center, radius0));

let radius1 = 1.5;
assert!(mvt.collides(&center, radius1));
```

`Mvt` is immutable once built. If you need to insert new points after construction, use
`MutableMvt` instead, which supports `MutableMvt::insert`/`MutableMvt::insert_points` at some
cost to memory usage and construction time; see `MutableMvt`'s documentation for details.

## Performance

The performance of the MVT is excellent, outpacing even the SIMD-accelerated [CAPT](https://github.com/KavrakiLab/capt).

![Plot of construction time throughput](https://github.com/claytonwramsey/mvtable/raw/HEAD/doc/mbm_throughput_construction.svg?sanitize=true)
![Plot of memory consumption throughput](https://github.com/claytonwramsey/mvtable/raw/HEAD/doc/mbm_throughput_memory.svg?sanitize=true)
![Plot of query time throughput](https://github.com/claytonwramsey/mvtable/raw/HEAD/doc/mbm_throughput_query.svg?sanitize=true)

In throughput benchmarks, we find that the MVT has superior query throughput on large point clouds to all other compared methods,
despite having cheap construction times and memory costs.

![Plot of planning time versus primitive-only baseline](https://github.com/claytonwramsey/mvtable/raw/HEAD/doc/primitive_vs_other.svg?sanitize=true)

On the whole, MVTs offer extremely good planning performance, on par with ground-truth primitive geometry.

## Optional features

Besides the default-enabled `std` feature, this crate exposes one opt-in feature, `simd`, which
enables a SIMD-parallel interface for querying `Mvt`s. The `simd` feature requires nightly Rust
and therefore should be considered unstable. This enables the function `Mvt::collides_simd`, a
parallel collision checker for batches of search queries.

## License

This project is dual-licensed under the
[MIT License](https://github.com/claytonwramsey/mvtable/blob/HEAD/LICENSE-MIT.md) and the
[Apache License, Version 2.0](https://github.com/claytonwramsey/mvtable/blob/HEAD/LICENSE-APACHE.md).
You may choose either license at your option.
