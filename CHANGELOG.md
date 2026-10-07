# Changelog

All notable changes to the `mvtable` crate are documented in this file.
This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Performance

- `Mvt` and `MutableMvt` now reserve their table storage once at construction and on each batch insert.
  This speeds up construction on large, sparse grids by up to 9.5x, but might not improve performance on realistic datasets.

## 0.1.1 - 2026-10-06

### New features

- Added `Mvt::r_point` and `MutableMvt::r_point`, which return the point radius passed at construction.
- Added `Mvt::voxel_width` and `MutableMvt::voxel_width`, which return the voxel width passed at construction.

## 0.1.0 - 2026-08-10

Initial release.
