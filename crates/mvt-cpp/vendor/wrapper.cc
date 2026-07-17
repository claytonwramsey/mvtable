// C ABI wrapper around the vendored `vamp::collision::MVT` (see
// vendor/README.md), so it can be called from Rust via FFI. Not part of the
// vendored upstream code.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
// `mvt.hh`'s `get_robot_name`/`record_mvt_state_to_file` (research/mvt branch,
// see vendor/README.md) use `<chrono>`/`<sstream>`/`<ctime>` without including
// them; `<memory>` for `std::unique_ptr` is missing on both branches.
#include <chrono>
#include <ctime>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <vamp/collision/mvt.hh>

using vamp::collision::MVT;
using vamp::collision::Point;

namespace {
std::vector<Point> to_points(const float *points_xyz, std::size_t n_points) {
    std::vector<Point> points(n_points);
    for (std::size_t i = 0; i < n_points; ++i) {
        points[i] = { points_xyz[3 * i], points_xyz[3 * i + 1], points_xyz[3 * i + 2] };
    }
    return points;
}

// Determine whether it is safe to query an MVT with a given sphere.
bool safe_to_query(const MVT *mvt, const Point &center, float radius) {
    // too big or outside aabb yield segfaults
    if (!(radius <= mvt->max_query_radius)) {
        return false;
    }
    for (int k = 0; k < 3; ++k) {
        if (center[k] + radius < mvt->workspace_aabb_min[k]
            || center[k] - radius > mvt->workspace_aabb_max[k]) {
            return false;
        }
    }
    return true;
}
} // namespace

extern "C" {
MVT *mvt_cpp_new(const float *points_xyz, std::size_t n_points, float min_radius, float max_radius,
    const float *workspace_min, const float *workspace_max, float point_radius) {
    auto points = to_points(points_xyz, n_points);
    Point wmin { workspace_min[0], workspace_min[1], workspace_min[2] };
    Point wmax { workspace_max[0], workspace_max[1], workspace_max[2] };
    return new MVT(points, min_radius, max_radius, wmin, wmax, point_radius);
}

void mvt_cpp_free(MVT *mvt) { delete mvt; }

bool mvt_cpp_collides(const MVT *mvt, const float *center, float radius) {
    Point c { center[0], center[1], center[2] };
    if (!safe_to_query(mvt, c, radius)) {
        return false;
    }
    return mvt->collides(c, radius);
}

// `centers_x`/`centers_y`/`centers_z`/`radii` each point to
// `mvt_cpp_simd_width()` floats.
bool mvt_cpp_collides_simd(const MVT *mvt, const float *centers_x, const float *centers_y,
    const float *centers_z, const float *radii) {
    const std::size_t width = MVT::FVectorT::num_scalars;
    bool all_safe = true;
    for (std::size_t lane = 0; lane < width; ++lane) {
        Point c { centers_x[lane], centers_y[lane], centers_z[lane] };
        if (!safe_to_query(mvt, c, radii[lane])) {
            all_safe = false;
            break;
        }
    }
    if (!all_safe) {
        // Falls out of true SIMD for this one batch (see `safe_to_query`'s comment)
        // - rare in practice, and correctness/not-crashing matters more here than
        // this batch's timing purity.
        bool result = false;
        for (std::size_t lane = 0; lane < width; ++lane) {
            Point c { centers_x[lane], centers_y[lane], centers_z[lane] };
            if (safe_to_query(mvt, c, radii[lane]) && mvt->collides(c, radii[lane])) {
                result = true;
                break;
            }
        }
        return result;
    }

    // `is_aligned = false`: unlike `mvt`'s own voxel coordinate pools (64-byte
    // aligned via `posix_memalign`), these pointers come straight from the caller
    // with no alignment guarantee, and `FVectorT`'s default single-pointer
    // constructor issues an aligned AVX load that segfaults on an unaligned
    // pointer.
    std::array<MVT::FVectorT, 3> centers = {
        MVT::FVectorT(centers_x, false),
        MVT::FVectorT(centers_y, false),
        MVT::FVectorT(centers_z, false),
    };
    MVT::FVectorT r(radii, false);
    return mvt->collides_simd(centers, r);
}

std::size_t mvt_cpp_simd_width() { return MVT::FVectorT::num_scalars; }

// Stack + heap memory used by `mvt`.
std::size_t mvt_cpp_memory_used(const MVT *mvt) {
    std::size_t total = sizeof(MVT);
    total += mvt->point_coord_pool_size * sizeof(float);
    total += mvt->hierarchy_pool_size_bytes;
    total += mvt->voxel_storage.capacity() * sizeof(MVT::Voxel);
    return total;
}

// Predicts whether `mvt_cpp_new` with these same
// arguments would hit `MVT`'s internal hierarchy-pool-exhaustion
// `std::terminate()`.
bool mvt_cpp_would_overflow(const float *points_xyz, std::size_t n_points, float max_radius,
    const float *workspace_min, const float *workspace_max) {
    if (n_points == 0) {
        return false;
    }

    const float workspace_width = workspace_max[0] - workspace_min[0];
    if (!(workspace_width > 0.0f) || !(max_radius > 0.0f)) {
        return true;
    }

    const std::uint32_t grid_width_u32
        = std::min(static_cast<std::uint32_t>(std::floor(workspace_width / max_radius)),
            static_cast<std::uint32_t>(std::numeric_limits<std::uint16_t>::max()));
    if (grid_width_u32 == 0) {
        return true;
    }
    const std::uint16_t grid_width = static_cast<std::uint16_t>(grid_width_u32);
    const float inverse_scale_factor = static_cast<float>(grid_width) / workspace_width;

    const std::size_t xy_column_capacity
        = static_cast<std::size_t>(static_cast<double>(grid_width) * grid_width * 0.8);

    std::unordered_map<std::uint32_t, bool> occupied_columns;
    for (std::size_t i = 0; i < n_points; ++i) {
        const float x = points_xyz[3 * i];
        const float y = points_xyz[3 * i + 1];
        const float vx = (x - workspace_min[0]) * inverse_scale_factor;
        const float vy = (y - workspace_min[1]) * inverse_scale_factor;
        const auto max_index = static_cast<float>(grid_width - 1);
        const auto ix = static_cast<std::uint16_t>(std::clamp(vx, 0.0f, max_index));
        const auto iy = static_cast<std::uint16_t>(std::clamp(vy, 0.0f, max_index));
        const std::uint32_t column_key
            = (static_cast<std::uint32_t>(ix) << 16) | static_cast<std::uint32_t>(iy);
        occupied_columns[column_key] = true;
    }
    return occupied_columns.size() > xy_column_capacity;
}
}
