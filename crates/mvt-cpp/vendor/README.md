# Vendored `vamp_mvt` headers

These headers are copied verbatim from
[chingchennn/vamp_mvt](https://github.com/chingchennn/vamp_mvt), the reference C++
implementation of the MVT described in the paper this repository implements
(https://rasevents.org/uploads/documents/pdfviewer/b1/d6/223112-5124.pdf). Licensed under
Apache-2.0 (see `LICENSE.txt`, also from that repository).

`vamp/collision/mvt.hh` is from the `research/mvt` branch at commit
`61db78d981585f9063bb4b0593d627db9a1ba5ee`, not `main` (commit `17f6d1c8fd730481dada8b69685136a46dc55bd8`,
where the rest of these headers still come from - they're byte-identical between the two
branches). `research/mvt` reworks `MVT`'s memory layout significantly: point coordinates are now
allocated in a two-phase count-then-allocate pass (`build_spatial_grid_two_phase`) sized exactly
to the real data, instead of `main`'s fixed per-voxel capacity estimate
(`(max_radius / 0.02)^3`) - which is what let a locally dense cluster of real-world points
overflow it (see the `Overflow`/`would_overflow` git history in `crates/mvt-cpp/src/lib.rs` for
what that looked like in practice). The X/Y/Z-level tables (`hierarchy_pool`, unified from
`main`'s separate `pointer_array_pool`/`voxel_index_pool`) are still sized from an occupancy
estimate, just a less conservative one (80% vs. 50%) - see `mvt_cpp_would_overflow` in
`wrapper.cc` for the capacity check that remains.

Only the subset of `vamp`'s headers needed to compile `vamp::collision::MVT` is vendored here:
the generic `VectorInterface` scaffolding plus both SIMD vector backends it dispatches to
(`vector/avx.hh` on x86_64, `vector/neon.hh` on AArch64 - selected by `vamp/vector.hh`'s
`#if defined(__x86_64__) / __ARM_NEON`), not anything from `vamp`'s planning, robot-model, or
Python-binding code, none of which `mvt.hh` depends on.

- `vamp/collision/mvt.hh` - the MVT itself.
- `vamp/collision/math.hh` - defines `Point` (`std::array<float, 3>`) and small scalar/vector
  math helpers `mvt.hh` uses.
- `vamp/vector.hh`, `vamp/vector/interface.hh`, `vamp/vector/avx.hh`, `vamp/vector/neon.hh`,
  `vamp/vector/utils.hh` - the generic SIMD vector wrapper (`FloatVector`/`IntVector`) and its
  AVX2 (8 lanes) and NEON (4 lanes) specializations. Only one of `avx.hh`/`neon.hh` is ever
  compiled in for a given target; `mvt_cpp::SIMD_WIDTH` on the Rust side tracks whichever one
  that build picked (see `build.rs`'s `cfg(target_arch = ...)`).
- `vamp/constants.hh`, `vamp/utils.hh` - small shared utilities these headers pull in.

Only `x86_64` (AVX2) and `aarch64` (NEON) are supported; `build.rs` fails fast on any other
target rather than silently picking one.
