# MVT: Multilevel Voxel Tables

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

A paper explaining the internals of the data structure is available at the
[ICRA conference proceedings](https://rasevents.org/uploads/documents/pdfviewer/b1/d6/223112-5124.pdf).

## Usage

The core data structure in this library is the `Mvt`, a sparse voxel grid used for
collision checking. `Mvt`s are polymorphic over dimension and floating-point type. On
construction, they take in a list of points in a point cloud and the maximum radius that will
be used for querying, which is used to size the grid's voxels.

```rust
use mvtable::Mvt;

// list of points in cloud
let points = [[0.0, 1.1], [0.2, 3.1]];
let r_max = 2.0;

let mvt = Mvt::<2>::new(&points, r_max);
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

## Performance

The performance of the MVT is excellent, outpacing even the SIMD-accelerated [CAPT](https://github.com/KavrakiLab/capt).

![doc/mbm_throughput.svg](Plot of collision-checking throughput)

In throughput benchmarks, we find that the MVT has superior query throughput on large point clouds to all other compared methods,
despite having cheap construction times and memory costs.

![doc/mbm_plan_times.svg](Plot of planning time distributions)

These performance gains translate to motion planning, where we can see a respectable constant-factor speedup from MVTs on most robots.

## Optional features

This crate exposes one feature, `simd`, which enables a SIMD-parallel interface for querying
`Mvt`s. The `simd` feature requires nightly Rust and therefore should be considered
unstable. This enables the function `Mvt::collides_simd`, a parallel collision checker for
batches of search queries.
