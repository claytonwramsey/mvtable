fn main() {
    println!("cargo::rerun-if-changed=vendor/wrapper.cc");
    println!("cargo::rerun-if-changed=vendor/vamp");

    let target_arch =
        std::env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH not set");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .include("vendor")
        .file("vendor/wrapper.cc")
        .opt_level(3)
        // Matches `cmake/CompilerSettings.cmake`'s `VAMP_FAST_ARGS` in the upstream repo (see
        // vendor/README.md), so the vendored code runs under the same floating-point relaxations
        // its own authors benchmark it with.
        .flag_if_supported("-fno-math-errno")
        .flag_if_supported("-fno-signed-zeros")
        .flag_if_supported("-fno-trapping-math")
        .flag_if_supported("-fno-rounding-math")
        .flag_if_supported("-ffp-contract=fast")
        .warnings(false);

    // Matches `cmake/CompilerSettings.cmake`'s per-architecture `VAMP_ARCH` (see
    // vendor/README.md): picks which of `vector/avx.hh` (x86_64)/`vector/neon.hh` (aarch64)
    // `vamp/vector.hh` compiles in, and must match `mvt_cpp::SIMD_WIDTH`'s own `cfg(target_arch)`
    // on the Rust side (8 lanes on x86_64/AVX2, 4 on aarch64/NEON).
    match target_arch.as_str() {
        "x86_64" => {
            build
                .flag_if_supported("-march=native")
                .flag_if_supported("-mavx2")
                // Upstream only enables this (and the Clang-specific flags after it) for x86_64.
                .flag_if_supported("-fassociative-math");
        }
        "aarch64" => {
            build
                .flag_if_supported("-mcpu=native")
                .flag_if_supported("-mtune=native")
                // Works around a GCC 13+ NEON vector-type-conversion error; harmless (and
                // skipped by `flag_if_supported`) on compilers that don't need or support it.
                .flag_if_supported("-flax-vector-conversions");
        }
        other => panic!(
            "mvt-cpp's vendored C++ (see vendor/README.md) only supports x86_64 (AVX2) and \
             aarch64 (NEON), not {other:?}"
        ),
    }

    build.compile("mvt_cpp_wrapper");
}
