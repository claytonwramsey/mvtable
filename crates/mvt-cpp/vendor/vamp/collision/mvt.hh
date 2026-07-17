#pragma once

#include <unistd.h>
#include <iostream>
#include <fstream> 
#include <iomanip>
#include <algorithm>
#include <cstdint>
#include <new>
#include <numeric>
#include <limits>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstring>

#include <vamp/collision/math.hh>
#include <vamp/vector.hh>

// #define RECORD_MEM
// #define BENCHMARK_CC
// #define PRINT_CC

namespace vamp::collision
{
    /**
     * Multi-level Voxel Table (MVT) - A hierarchical spatial data structure
     * for efficient collision detection using a three-level sparse table.
     * 
     * Assumption: all points to be inserted to MVT is in the given workspace bounds 
     */
    struct MVT
    {
        // ====================================================================
        // TYPE DEFINITIONS
        // ====================================================================
        
        using FVectorT = FloatVector<>;
        using IVectorT = IntVector<>;
        using VoxelIndex = uint32_t;
        
        static constexpr VoxelIndex INVALID_VOXEL_INDEX = std::numeric_limits<VoxelIndex>::max();
        static constexpr uint16_t MAX_GRID_WIDTH = std::numeric_limits<uint16_t>::max();
        
        // Three-level table types
        using TableOffset = uint32_t;
        using ZLevelTable = VoxelIndex*;   
        using YLevelTable = TableOffset*;  
        using XLevelTable = TableOffset*;

        // ====================================================================
        // VOXEL STRUCTURE
        // ====================================================================
        
        struct alignas(32) Voxel {
            float* x_coords = nullptr;
            float* y_coords = nullptr;
            float* z_coords = nullptr;
            size_t point_count = 0;
            size_t capacity = 0;
            
            Point bbox_min = {0.0f, 0.0f, 0.0f};
            Point bbox_max = {0.0f, 0.0f, 0.0f};
            
            Voxel() = default;

            void initialize_with_pool(float* x_ptr, float* y_ptr, float* z_ptr, size_t cap) {
                x_coords = x_ptr;
                y_coords = y_ptr;
                z_coords = z_ptr;
                capacity = cap;
                point_count = 0;
            }

            void add_point(const Point& point, const float& point_radius) {
                if (point_count >= capacity) {
                    std::cout << "Try to add " << point_count + 1 << "th point to a voxel" << std::endl;
                    throw std::runtime_error("Voxel capacity exceeded");
                }

                x_coords[point_count] = point[0];
                y_coords[point_count] = point[1];
                z_coords[point_count] = point[2];
                
                update_bounding_box(point, point_radius);
                ++point_count;
            }

        private:
            void update_bounding_box(const Point& point, const float& point_radius) {
                float px_min = point[0] - point_radius;
                float py_min = point[1] - point_radius;
                float pz_min = point[2] - point_radius;
                float px_max = point[0] + point_radius;
                float py_max = point[1] + point_radius;
                float pz_max = point[2] + point_radius;

                if (point_count == 0) {
                    bbox_min = {px_min, py_min, pz_min};
                    bbox_max = {px_max, py_max, pz_max};
                } else {
                    bbox_min[0] = std::min(bbox_min[0], px_min);
                    bbox_min[1] = std::min(bbox_min[1], py_min);
                    bbox_min[2] = std::min(bbox_min[2], pz_min);
                    bbox_max[0] = std::max(bbox_max[0], px_max);
                    bbox_max[1] = std::max(bbox_max[1], py_max);
                    bbox_max[2] = std::max(bbox_max[2], pz_max);
                }
            }
        };

        // ====================================================================
        // MEMBER VARIABLES
        // ====================================================================
        
        // Query parameters
        float min_query_radius;
        float max_query_radius;
        float point_radius;
        
        // Spatial bounds
        Point workspace_aabb_min;
        Point workspace_aabb_max;
        Point global_aabb_min;
        Point global_aabb_max;
        
        // Grid configuration
        float inverse_scale_factor;
        float voxel_size;
        uint16_t grid_width;
        uint16_t table_array_len;
        uint16_t z_table_array_len;
        
        // Memory pools
        std::unique_ptr<float[], decltype(&std::free)> point_coord_pool{nullptr, &std::free};
        size_t point_coord_pool_size = 0;
        size_t point_coord_pool_used = 0;
        size_t estimated_max_point_per_voxel = 0;
        
        static constexpr TableOffset NULL_OFFSET = std::numeric_limits<TableOffset>::max();
        std::unique_ptr<uint8_t[], decltype(&std::free)> hierarchy_pool{nullptr, &std::free};
        size_t hierarchy_pool_size_bytes = 0;
        size_t hierarchy_pool_used_bytes = 0;
        
        // Voxel storage and hierarchy entry
        std::vector<Voxel> voxel_storage;
        XLevelTable x_level_table = nullptr;
        
        // SIMD-optimized bounds
        FVectorT simd_global_min_x, simd_global_min_y, simd_global_min_z;
        FVectorT simd_global_max_x, simd_global_max_y, simd_global_max_z;
        FVectorT simd_workspace_min_x, simd_workspace_min_y, simd_workspace_min_z;

        // ====================================================================
        // CONSTRUCTOR & DESTRUCTOR
        // ====================================================================
        
        MVT(const std::vector<Point>& points,
            const float min_radius,
            const float max_radius,
            const Point &workspace_aabb_min, 
            const Point &workspace_aabb_max, 
            const float point_radius) noexcept
            : min_query_radius{min_radius},
              max_query_radius{max_radius},
              workspace_aabb_min{workspace_aabb_min},
              workspace_aabb_max{workspace_aabb_max},
              point_radius{point_radius}
        {
            if (points.empty()) {
                initialize_empty_bounds();
                return;
            }
            
            configure_grid();
            initialize_hierarchy_pool();
            initialize_voxel_storage();
            build_spatial_grid_two_phase(points);
            compute_global_bounds();
            setup_simd_vectors();

#ifdef BENCHMARK_CC
            benchmark_collision_queries("../nanoflann_dataset/cage_fetch_capt_q/60/collide.txt");
            benchmark_collision_queries("../nanoflann_dataset/cage_fetch_capt_q/60/safe.txt");
#endif
#ifdef RECORD_MEM
            record_mvt_state_to_file("scripts/log/");
#endif
        }

        MVT(const MVT& other)
            : min_query_radius(other.min_query_radius),
              max_query_radius(other.max_query_radius),
              point_radius(other.point_radius),
              workspace_aabb_min(other.workspace_aabb_min),
              workspace_aabb_max(other.workspace_aabb_max),
              global_aabb_min(other.global_aabb_min),
              global_aabb_max(other.global_aabb_max),
              inverse_scale_factor(other.inverse_scale_factor),
              voxel_size(other.voxel_size),
              grid_width(other.grid_width),
              table_array_len(other.table_array_len),
              z_table_array_len(other.z_table_array_len),
              point_coord_pool_size(other.point_coord_pool_size),
              point_coord_pool_used(other.point_coord_pool_used),
              estimated_max_point_per_voxel(other.estimated_max_point_per_voxel),
              hierarchy_pool_size_bytes(other.hierarchy_pool_size_bytes),
              hierarchy_pool_used_bytes(other.hierarchy_pool_used_bytes),
              voxel_storage(other.voxel_storage)
        {
            copy_memory_pools(other);
            update_pointers_after_copy(other);
            setup_simd_vectors();
        }

        ~MVT() = default;

        // ====================================================================
        // COLLISION DETECTION
        // ====================================================================
        
        // Scalar collision detection
        [[nodiscard]] auto collides(const Point& center, float radius) const noexcept -> bool
        {
            const float query_radius = radius + point_radius;
            const float query_radius_squared = query_radius * query_radius;

            // Early exit: Global AABB check
            if (center[0] + query_radius < global_aabb_min[0] ||
                center[0] - query_radius > global_aabb_max[0] ||
                center[1] + query_radius < global_aabb_min[1] ||
                center[1] - query_radius > global_aabb_max[1] ||
                center[2] + query_radius < global_aabb_min[2] ||
                center[2] - query_radius > global_aabb_max[2]) {
                return false;
            }

            // Compute grid space coordinates and query bounds
            const float grid_query_radius = std::min(1.0f, query_radius * inverse_scale_factor);
            const float grid_center_x_float = (center[0] - workspace_aabb_min[0]) * inverse_scale_factor;
            const float grid_center_y_float = (center[1] - workspace_aabb_min[1]) * inverse_scale_factor;
            const float grid_center_z_float = (center[2] - workspace_aabb_min[2]) * inverse_scale_factor;
            
            //Calculate voxel iteration bounds
            const uint16_t min_x = static_cast<uint16_t>(std::max(0.0f, (grid_center_x_float - grid_query_radius)));
            const uint16_t max_x = static_cast<uint16_t>(std::min(static_cast<float>(grid_width - 1), (grid_center_x_float + grid_query_radius)));
            const uint16_t min_y = static_cast<uint16_t>(std::max(0.0f, (grid_center_y_float - grid_query_radius)));
            const uint16_t max_y = static_cast<uint16_t>(std::min(static_cast<float>(grid_width - 1), (grid_center_y_float + grid_query_radius)));
            const uint16_t min_z = static_cast<uint16_t>(std::max(0.0f, (grid_center_z_float - grid_query_radius)));
            const uint16_t max_z = static_cast<uint16_t>(std::min(static_cast<float>(grid_width - 1), (grid_center_z_float + grid_query_radius)));

            const uint8_t* hierarchy_base = hierarchy_pool.get();
            // Traverse three-level spatial hierarchy
            for (uint16_t voxel_x = min_x; voxel_x <= max_x; ++voxel_x) {
                if (x_level_table[voxel_x] == NULL_OFFSET) continue;
                const TableOffset* y_table = reinterpret_cast<const TableOffset*>(hierarchy_base + x_level_table[voxel_x]);

                for (uint16_t voxel_y = min_y; voxel_y <= max_y; ++voxel_y) {
                    if (y_table[voxel_y] == NULL_OFFSET) continue;
                    const VoxelIndex* z_table = reinterpret_cast<const VoxelIndex*>(hierarchy_base + y_table[voxel_y]);
                    
                    for (uint16_t voxel_z = min_z; voxel_z <= max_z; ++voxel_z) {
                        VoxelIndex voxel_index = z_table[voxel_z];
                        if (voxel_index == INVALID_VOXEL_INDEX) continue;
                        
                        const Voxel& voxel = voxel_storage[voxel_index];
                        
                        // Voxel-level AABB culling
                        if (center[0] + query_radius < voxel.bbox_min[0] ||
                            center[0] - query_radius > voxel.bbox_max[0] ||
                            center[1] + query_radius < voxel.bbox_min[1] ||
                            center[1] - query_radius > voxel.bbox_max[1] ||
                            center[2] + query_radius < voxel.bbox_min[2] ||
                            center[2] - query_radius > voxel.bbox_max[2]) {
                            continue;
                        }
                        
                        // Point-level collision detection
                        const size_t num_points = voxel.point_count;
                        for (size_t i = 0; i < num_points; ++i) {
                            const float dx = center[0] - voxel.x_coords[i];
                            const float dy = center[1] - voxel.y_coords[i];
                            const float dz = center[2] - voxel.z_coords[i];
                            const float distance_squared = dx * dx + dy * dy + dz * dz;
                            
                            if (distance_squared <= query_radius_squared) {
                                return true;
                            }
                        }
                    }
                }
            }
            
            return false;
        }

        // SIMD vectorized collision detection for multiple spheres
        auto inline collides_simd(const std::array<FVectorT, 3> &centers, 
                                FVectorT radii) const noexcept -> bool
        {
#ifdef PRINT_CC
            print_simd_args(centers, radii);
#endif
            constexpr size_t SIMD_WIDTH = FVectorT::num_scalars;

            // Compute query radii for all spheres
            const FVectorT point_radius_vec = FVectorT::fill(point_radius);
            const FVectorT query_radii = radii + point_radius_vec;

            // SIMD global AABB check - cull entire lanes that are completely outside
            const auto outside_x_low = (centers[0]) < (simd_global_min_x - query_radii);
            const auto outside_x_high = (simd_global_max_x + query_radii) < (centers[0]);
            const auto outside_y_low = (centers[1]) < (simd_global_min_y - query_radii);
            const auto outside_y_high = (simd_global_max_y + query_radii) < (centers[1]);
            const auto outside_z_low = (centers[2]) < (simd_global_min_z - query_radii);
            const auto outside_z_high = (simd_global_max_z + query_radii) < (centers[2]);
            
            const auto outside_mask = outside_x_low | outside_x_high | 
                                    outside_y_low | outside_y_high | 
                                    outside_z_low | outside_z_high;
            
            if (outside_mask.all()) {
                return false;  // All spheres are outside global bounds
            }
            
            // Transform centers to grid space
            const FVectorT inv_scale = FVectorT::fill(inverse_scale_factor);
            const FVectorT grid_center_x = (centers[0] - simd_workspace_min_x) * inv_scale;
            const FVectorT grid_center_y = (centers[1] - simd_workspace_min_y) * inv_scale;
            const FVectorT grid_center_z = (centers[2] - simd_workspace_min_z) * inv_scale;
            const auto query_radii_squared = query_radii * query_radii;

            // Extract scalar arrays for per-sphere processing
            const auto centers_x_array = centers[0].to_array();
            const auto centers_y_array = centers[1].to_array();
            const auto centers_z_array = centers[2].to_array();
            const auto query_radii_array = query_radii.to_array();
            const auto query_radii_squared_array = query_radii_squared.to_array();
            const auto outside_array = outside_mask.to_array();
            const auto grid_x_array = grid_center_x.to_array();
            const auto grid_y_array = grid_center_y.to_array();
            const auto grid_z_array = grid_center_z.to_array();
            
            const uint8_t* hierarchy_base = hierarchy_pool.get();
            // Process each sphere individually
            for (size_t sphere_idx = 0; sphere_idx < SIMD_WIDTH; ++sphere_idx) {
                // Skip spheres that failed global AABB test
                if (outside_array[sphere_idx] != 0) {
                    continue;
                }
                
                // Extract sphere parameters
                const Point center = {centers_x_array[sphere_idx], centers_y_array[sphere_idx], centers_z_array[sphere_idx]};
                const float query_radius = query_radii_array[sphere_idx];
                const float query_radius_squared = query_radii_squared_array[sphere_idx];
                const float grid_query_radius = query_radius * inverse_scale_factor;
                // const float grid_query_radius = std::min(1.0f, query_radius * inverse_scale_factor);
                const float grid_center_x_float = grid_x_array[sphere_idx];
                const float grid_center_y_float = grid_y_array[sphere_idx];
                const float grid_center_z_float = grid_z_array[sphere_idx];
                
                // Calculate voxel iteration bounds for this sphere
                const float max_grid_idx_float = static_cast<float>(grid_width - 1);
                const uint16_t min_x = static_cast<uint16_t>(std::max(0.0f, (grid_center_x_float - grid_query_radius)));
                const uint16_t max_x = static_cast<uint16_t>(std::min(max_grid_idx_float, (grid_center_x_float + grid_query_radius)));
                const uint16_t min_y = static_cast<uint16_t>(std::max(0.0f, (grid_center_y_float - grid_query_radius)));
                const uint16_t max_y = static_cast<uint16_t>(std::min(max_grid_idx_float, (grid_center_y_float + grid_query_radius)));
                const uint16_t min_z = static_cast<uint16_t>(std::max(0.0f, (grid_center_z_float - grid_query_radius)));
                const uint16_t max_z = static_cast<uint16_t>(std::min(max_grid_idx_float, (grid_center_z_float + grid_query_radius)));

                // Traverse spatial hierarchy for this sphere
                for (uint16_t voxel_x = min_x; voxel_x <= max_x; ++voxel_x) {
                    // Level 1: Get Y-level table offset from X-level table
                    TableOffset y_offset = x_level_table[voxel_x];
                    if (y_offset == NULL_OFFSET) continue;
                    // Resolve the actual pointer by adding the offset to the hierarchy base
                    const TableOffset* y_level_table = reinterpret_cast<const TableOffset*>(hierarchy_base + y_offset);
                    
                    for (uint16_t voxel_y = min_y; voxel_y <= max_y; ++voxel_y) {
                        // Level 2: Get Z-level (voxel index) table offset from Y-level table
                        TableOffset z_offset = y_level_table[voxel_y];
                        if (z_offset == NULL_OFFSET) continue;
                        
                        // Resolve the actual pointer for the Z-level table
                        const VoxelIndex* z_level_table = reinterpret_cast<const VoxelIndex*>(hierarchy_base + z_offset);
                        
                        for (uint16_t voxel_z = min_z; voxel_z <= max_z; ++voxel_z) {
                            VoxelIndex voxel_index = z_level_table[voxel_z];
                            if (voxel_index == INVALID_VOXEL_INDEX) continue;
                            
                            const Voxel& voxel = voxel_storage[voxel_index];
                            
                            // Voxel-level AABB culling
                            if (center[0] + query_radius < voxel.bbox_min[0] ||
                                center[0] - query_radius > voxel.bbox_max[0] ||
                                center[1] + query_radius < voxel.bbox_min[1] ||
                                center[1] - query_radius > voxel.bbox_max[1] ||
                                center[2] + query_radius < voxel.bbox_min[2] ||
                                center[2] - query_radius > voxel.bbox_max[2]) {
                                continue;
                            }
                            
                            // SIMD point-level collision detection
                            const size_t num_points = voxel.point_count;
                            const auto* x_coords = voxel.x_coords;
                            const auto* y_coords = voxel.y_coords;
                            const auto* z_coords = voxel.z_coords;

                            const FVectorT sphere_x = FVectorT::fill(center[0]);
                            const FVectorT sphere_y = FVectorT::fill(center[1]);
                            const FVectorT sphere_z = FVectorT::fill(center[2]);
                            const FVectorT sphere_radius_sq = FVectorT::fill(query_radius_squared);

                            // Process points in SIMD chunks
                            for (size_t point_idx = 0; point_idx < num_points; point_idx += SIMD_WIDTH) {
                                const FVectorT point_x(x_coords + point_idx);
                                const FVectorT point_y(y_coords + point_idx);
                                const FVectorT point_z(z_coords + point_idx);
                                
                                const FVectorT dx = sphere_x - point_x;
                                const FVectorT dy = sphere_y - point_y;
                                const FVectorT dz = sphere_z - point_z;
                                const FVectorT dist_sq = dx * dx + dy * dy + dz * dz;
                                
                                const auto collision_mask = dist_sq <= sphere_radius_sq;
                                if (collision_mask.any()) {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
            
            return false;

            // // Call serial collision checking
            // constexpr size_t SIMD_WIDTH = FVectorT::num_scalars;
            // const auto centers_x = centers[0].to_array();
            // const auto centers_y = centers[1].to_array();
            // const auto centers_z = centers[2].to_array();
            // const auto radiivec = radii.to_array();

            // for (size_t i = 0; i < SIMD_WIDTH; ++i) {
            //     Point center = {centers_x[i], centers_y[i], centers_z[i]};
            //     float radius = radiivec[i];

            //     if (collides(center, radius)) {
            //         return true; // Early exit if any sphere collides
            //     }
            // }

            // return false; // No collisions detected
        }

    private:
        // ====================================================================
        // INITIALIZATION HELPERS
        // ====================================================================
        
        void initialize_empty_bounds() {
            constexpr float max_val = std::numeric_limits<float>::max();
            constexpr float min_val = std::numeric_limits<float>::lowest();
            
            global_aabb_min = Point{max_val, max_val, max_val};
            global_aabb_max = Point{min_val, min_val, min_val};
        }

        void configure_grid() {
            const float workspace_width = workspace_aabb_max[0] - workspace_aabb_min[0];
            voxel_size = (max_query_radius) + point_radius;
            
            grid_width = static_cast<uint16_t>(std::min(
                static_cast<uint32_t>(std::floor(workspace_width / voxel_size)), // Empirically < 100 for manipulator robots
                static_cast<uint32_t>(MAX_GRID_WIDTH) // upper bound
            ));

            inverse_scale_factor = grid_width / workspace_width;
        }

        void initialize_point_coord_pool() {
            // 1. Calculate capacity per voxel (aligned to SIMD lanes)
            auto align_to_simd = [](size_t n) {
                constexpr size_t WIDTH = FVectorT::num_scalars;
                return ((n + WIDTH - 1) / WIDTH) * WIDTH;
            };
        
            // Estimation
            size_t raw_estimate = static_cast<size_t>(std::pow(max_query_radius / 0.02f, 3.0f));
            estimated_max_point_per_voxel = align_to_simd(raw_estimate);
        
            // 2. Calculate Total Pool Size
            size_t total_voxels = static_cast<size_t>(grid_width) * grid_width * grid_width;
            
            // Assume 10% occupancy and need 3 arrays (X, Y, Z) per voxel
            double sparsity_factor = 0.1;
            point_coord_pool_size = static_cast<size_t>(
                total_voxels * sparsity_factor * estimated_max_point_per_voxel * 3
            );
        
            // 3. Physical Allocation
            size_t total_bytes = point_coord_pool_size * sizeof(float);
            void* raw_ptr = nullptr;
            
            // Aligned to 64 bytes
            if (posix_memalign(&raw_ptr, 64, total_bytes) != 0) {
                throw std::runtime_error("Failed to allocate " + std::to_string(total_bytes) + " bytes");
            }
            
            point_coord_pool.reset(static_cast<float*>(raw_ptr));
            point_coord_pool_used = 0; // Ensure reset
        }

        /**
         * Initializes a unified memory pool for the X, Y, and Z level tables.
         * Uses 32-bit offsets to save space and simplify memory relocation.
         */
        void initialize_hierarchy_pool() {
            // 1. Estimate Pointer Table requirements (X and Y levels)
            const size_t estimated_xy_tables = 1 + grid_width;
            const size_t pointer_pool_bytes = estimated_xy_tables * grid_width * sizeof(TableOffset);
            table_array_len = grid_width;

            // 2. Estimate Voxel Index Table requirements (Z level)
            const size_t estimated_z_tables = static_cast<size_t>(grid_width) * grid_width * 0.8;
            const size_t z_table_pool_bytes = estimated_z_tables * grid_width * sizeof(VoxelIndex);
            z_table_array_len = grid_width;

            // 3. Calculate Total Size
            hierarchy_pool_size_bytes = pointer_pool_bytes + z_table_pool_bytes;
            hierarchy_pool_used_bytes = 0;

            // 4. Allocate Aligned Memory
            void* raw_ptr = nullptr;
            if (posix_memalign(&raw_ptr, 64, hierarchy_pool_size_bytes) != 0) {
                throw std::runtime_error("Failed to allocate hierarchy pool");
            }
            
            hierarchy_pool.reset(static_cast<uint8_t*>(raw_ptr));
        }

        void initialize_voxel_storage() {
            const size_t estimated_voxel_count = 
                static_cast<size_t>(grid_width) * grid_width * grid_width * 0.1;
            voxel_storage.reserve(estimated_voxel_count);
        }

        void setup_simd_vectors() {
            simd_global_min_x = FVectorT::fill(global_aabb_min[0]);
            simd_global_min_y = FVectorT::fill(global_aabb_min[1]);
            simd_global_min_z = FVectorT::fill(global_aabb_min[2]);
            simd_global_max_x = FVectorT::fill(global_aabb_max[0]);
            simd_global_max_y = FVectorT::fill(global_aabb_max[1]);
            simd_global_max_z = FVectorT::fill(global_aabb_max[2]);
            
            simd_workspace_min_x = FVectorT::fill(workspace_aabb_min[0]);
            simd_workspace_min_y = FVectorT::fill(workspace_aabb_min[1]);
            simd_workspace_min_z = FVectorT::fill(workspace_aabb_min[2]);
        }

        // ====================================================================
        // SPATIAL GRID CONSTRUCTION
        // ====================================================================
        
        void build_spatial_grid(const std::vector<Point>& points) {
            // Initialize root of three-level table hierarchy
            TableOffset x_table_offset;
            x_level_table = allocate_table<XLevelTable>(x_table_offset);

            // Insert each point into the corresponding voxel
            for (const auto& point : points) {
                // Transform point coordinates to grid space
                const float voxel_x_float = (point[0] - workspace_aabb_min[0]) * inverse_scale_factor;
                const float voxel_y_float = (point[1] - workspace_aabb_min[1]) * inverse_scale_factor;
                const float voxel_z_float = (point[2] - workspace_aabb_min[2]) * inverse_scale_factor;

                // Clamp to valid grid indices just in case
                const uint16_t voxel_x = static_cast<uint16_t>(std::clamp(voxel_x_float, 0.0f, static_cast<float>(grid_width - 1)));
                const uint16_t voxel_y = static_cast<uint16_t>(std::clamp(voxel_y_float, 0.0f, static_cast<float>(grid_width - 1)));
                const uint16_t voxel_z = static_cast<uint16_t>(std::clamp(voxel_z_float, 0.0f, static_cast<float>(grid_width - 1)));
                // Intervals are half-open [lower, upper) except for the last voxel

                // Level 1: Get Y-level
                TableOffset y_offset = x_level_table[voxel_x];
                if (y_offset == NULL_OFFSET) {
                    allocate_table<YLevelTable>(x_level_table[voxel_x]);
                }
                YLevelTable y_level_table = reinterpret_cast<YLevelTable>(hierarchy_pool.get() + x_level_table[voxel_x]);

                // Level 2: Get Z-level
                TableOffset z_offset = y_level_table[voxel_y];
                if (z_offset == NULL_OFFSET) {
                    allocate_table<ZLevelTable>(y_level_table[voxel_y]);
                }
                ZLevelTable z_level_table = reinterpret_cast<ZLevelTable>(hierarchy_pool.get() + y_level_table[voxel_y]);
                
                // Level 3: Get or create voxel
                VoxelIndex voxel_index = z_level_table[voxel_z];
                if (voxel_index == INVALID_VOXEL_INDEX) {
                    // Check capacity before adding to prevent reallocation during insertion
                    if (voxel_storage.size() >= voxel_storage.capacity()) {
                        std::cout << "Voxel storage capacity exceeded. Please consider reserving larger space." << std::endl;
                    }
                    
                    // Create new voxel and assign index
                    voxel_index = static_cast<VoxelIndex>(voxel_storage.size());
                    voxel_storage.emplace_back();
                    z_level_table[voxel_z] = voxel_index;
                    
                    // Allocate coordinate storage from memory pool
                    float* x_ptr = allocate_coords(estimated_max_point_per_voxel);
                    float* y_ptr = allocate_coords(estimated_max_point_per_voxel);
                    float* z_ptr = allocate_coords(estimated_max_point_per_voxel);
                    voxel_storage[voxel_index].initialize_with_pool(x_ptr, y_ptr, z_ptr, estimated_max_point_per_voxel);
                }
                
                // Add point to voxel
                voxel_storage[voxel_index].add_point(point, point_radius);
            }
        }
        
        void build_spatial_grid_two_phase(const std::vector<Point>& points) {
            // --- PHASE 1: INIT HIERARCHY &  ---
            
            // Initialize root of three-level table hierarchy
            TableOffset x_table_offset;
            x_level_table = allocate_table<XLevelTable>(x_table_offset);

            // Keep track of which voxel each point belongs to so we don't re-calculate in Phase 2
            std::vector<VoxelIndex> point_to_voxel(points.size());
            
            // Insert each point into the corresponding voxel
            for (size_t i = 0; i < points.size(); ++i) {
                const auto& point = points[i];
                // Transform point coordinates to grid space
                const float voxel_x_float = (point[0] - workspace_aabb_min[0]) * inverse_scale_factor;
                const float voxel_y_float = (point[1] - workspace_aabb_min[1]) * inverse_scale_factor;
                const float voxel_z_float = (point[2] - workspace_aabb_min[2]) * inverse_scale_factor;

                // Clamp to valid grid indices just in case
                const uint16_t voxel_x = static_cast<uint16_t>(std::clamp(voxel_x_float, 0.0f, static_cast<float>(grid_width - 1)));
                const uint16_t voxel_y = static_cast<uint16_t>(std::clamp(voxel_y_float, 0.0f, static_cast<float>(grid_width - 1)));
                const uint16_t voxel_z = static_cast<uint16_t>(std::clamp(voxel_z_float, 0.0f, static_cast<float>(grid_width - 1)));
                // Intervals are half-open [lower, upper) except for the last voxel

                // Level 1: Get Y-level
                TableOffset y_offset = x_level_table[voxel_x];
                if (y_offset == NULL_OFFSET) {
                    allocate_table<YLevelTable>(x_level_table[voxel_x]);
                }
                YLevelTable y_level_table = reinterpret_cast<YLevelTable>(hierarchy_pool.get() + x_level_table[voxel_x]);

                // Level 2: Get Z-level
                TableOffset z_offset = y_level_table[voxel_y];
                if (z_offset == NULL_OFFSET) {
                    allocate_table<ZLevelTable>(y_level_table[voxel_y]);
                }
                ZLevelTable z_level_table = reinterpret_cast<ZLevelTable>(hierarchy_pool.get() + y_level_table[voxel_y]);
                
                // Level 3: Get or create voxel
                VoxelIndex voxel_index = z_level_table[voxel_z];
                if (voxel_index == INVALID_VOXEL_INDEX) {
                    // Create new voxel and assign index
                    voxel_index = static_cast<VoxelIndex>(voxel_storage.size());
                    voxel_storage.emplace_back();
                    z_level_table[voxel_z] = voxel_index;
                }
                voxel_storage[voxel_index].point_count++; // Increment count only
                point_to_voxel[i] = voxel_index;          // Cache index
            }
                

            // --- PHASE 2: EXACT ALLOCATION & FILLING ---
            size_t total_required_floats = 0;
            constexpr size_t SIMD_WIDTH = FVectorT::num_scalars;

            for (auto& voxel : voxel_storage) {
                // Round each voxel's count up to SIMD width for alignment
                voxel.capacity = (voxel.point_count + SIMD_WIDTH - 1) & ~(SIMD_WIDTH - 1);
                total_required_floats += voxel.capacity * 3; // X, Y, Z
            }
            // Now call a modified version of initialize_point_coord_pool(total_required_floats)
            allocate_exact_point_pool(total_required_floats);

            // Assign pointers within the pool
            for (auto& voxel : voxel_storage) {
                voxel.x_coords = allocate_coords(voxel.capacity);
                voxel.y_coords = allocate_coords(voxel.capacity);
                voxel.z_coords = allocate_coords(voxel.capacity);
                // Reset count to 0 so add_point() can fill it correctly
                voxel.point_count = 0; 
            }

            for (size_t i = 0; i < points.size(); ++i) {
                VoxelIndex voxel_index = point_to_voxel[i];
                voxel_storage[voxel_index].add_point(points[i], point_radius);
            }
        }

        void compute_global_bounds() {
            initialize_empty_bounds();
            
            for (const auto& voxel : voxel_storage) {
                if (voxel.point_count > 0) {
                    global_aabb_min[0] = std::min(global_aabb_min[0], voxel.bbox_min[0]);
                    global_aabb_min[1] = std::min(global_aabb_min[1], voxel.bbox_min[1]);
                    global_aabb_min[2] = std::min(global_aabb_min[2], voxel.bbox_min[2]);
                    global_aabb_max[0] = std::max(global_aabb_max[0], voxel.bbox_max[0]);
                    global_aabb_max[1] = std::max(global_aabb_max[1], voxel.bbox_max[1]);
                    global_aabb_max[2] = std::max(global_aabb_max[2], voxel.bbox_max[2]);
                }
            }
        }

        // ====================================================================
        // MEMORY POOL ALLOCATION
        // ====================================================================
        
        void allocate_exact_point_pool(size_t total_floats) {
            point_coord_pool_size = total_floats;
            size_t total_bytes = point_coord_pool_size * sizeof(float);
            
            void* raw_ptr = nullptr;
            if (posix_memalign(&raw_ptr, 64, total_bytes) != 0) {
                throw std::runtime_error("Failed to allocate exact pool");
            }
            
            point_coord_pool.reset(static_cast<float*>(raw_ptr));
            point_coord_pool_used = 0;
            
            // Optional: Fill with Infinity for SIMD safety padding
            std::fill(point_coord_pool.get(), point_coord_pool.get() + point_coord_pool_size, 
                      std::numeric_limits<float>::infinity());
        }

        template<typename T>
        T allocate_table(TableOffset& out_offset) {
            const size_t element_size = std::is_same_v<T, ZLevelTable> ? sizeof(VoxelIndex) : sizeof(TableOffset);
            const size_t size_bytes = grid_width * element_size;
            
            if (hierarchy_pool_used_bytes + size_bytes > hierarchy_pool_size_bytes) {
                std::cout << "try to allocate " << hierarchy_pool_used_bytes + size_bytes << " bytes. Capacity: " << hierarchy_pool_size_bytes << "bytes" << std::endl;
                throw std::runtime_error("hierarchy pool exhausted");
            }
            
            out_offset = static_cast<TableOffset>(hierarchy_pool_used_bytes);
            T result = reinterpret_cast<T>(hierarchy_pool.get() + hierarchy_pool_used_bytes);
            
            if constexpr (std::is_same_v<T, ZLevelTable>) {
                std::fill(result, result + grid_width, INVALID_VOXEL_INDEX);
            } else {
                std::fill(result, result + grid_width, NULL_OFFSET);
            }
            
            hierarchy_pool_used_bytes += size_bytes;
            return result;
        }

        float* allocate_coords(size_t count) {
            if (point_coord_pool_used + count > point_coord_pool_size) {
                throw std::runtime_error("Point coordinate pool exhausted");
            }
            
            float* result = point_coord_pool.get() + point_coord_pool_used;
            point_coord_pool_used += count;
            return result;
        }

        // ====================================================================
        // COPY CONSTRUCTOR HELPERS
        // ====================================================================
        
        void copy_memory_pools(const MVT& other) {
            copy_point_coord_pool(other);
            copy_hierarchy_pool(other);
        }

        void copy_point_coord_pool(const MVT& other) {
            if (!other.point_coord_pool || other.point_coord_pool_size == 0) return;
            
            void* raw_ptr = nullptr;

            if (posix_memalign(&raw_ptr, 64, point_coord_pool_size * sizeof(float)) != 0) {
                throw std::runtime_error("Failed to allocate aligned memory pool");
            }
            
            point_coord_pool.reset(static_cast<float*>(raw_ptr));
            std::memcpy(point_coord_pool.get(), other.point_coord_pool.get(), 
                       point_coord_pool_used * sizeof(float));
        }

        void copy_hierarchy_pool(const MVT& other) {
            if (!other.hierarchy_pool || other.hierarchy_pool_size_bytes == 0) return;
            
            void* raw_ptr = nullptr;
            // Allocate the same amount of memory as the original pool
            if (posix_memalign(&raw_ptr, 64, other.hierarchy_pool_size_bytes) != 0) {
                throw std::runtime_error("Failed to allocate aligned hierarchy pool during copy");
            }
            
            hierarchy_pool.reset(static_cast<uint8_t*>(raw_ptr));
            
            // Copy the actual data (the X, Y, and Z table offsets)
            std::memcpy(hierarchy_pool.get(), other.hierarchy_pool.get(), 
                       other.hierarchy_pool_used_bytes);
        }

        void update_pointers_after_copy(const MVT& other) {
            relocate_table_hierarchy(other);
            relocate_voxel_coordinates(other);
        }

        void relocate_table_hierarchy(const MVT& other) {
            ptrdiff_t x_offset_in_pool = reinterpret_cast<uint8_t*>(other.x_level_table) - 
                                         other.hierarchy_pool.get();
            
            x_level_table = reinterpret_cast<XLevelTable>(hierarchy_pool.get() + x_offset_in_pool);
            // All internal offsets remain valid.
        }

        void relocate_voxel_coordinates(const MVT& other) {
            for (size_t idx = 0; idx < voxel_storage.size(); ++idx) {
                auto& voxel = voxel_storage[idx];
                const auto& other_voxel = other.voxel_storage[idx];
                
                if (other_voxel.x_coords != nullptr) {
                    voxel.x_coords = point_coord_pool.get() + (other_voxel.x_coords - other.point_coord_pool.get());
                    voxel.y_coords = point_coord_pool.get() + (other_voxel.y_coords - other.point_coord_pool.get());
                    voxel.z_coords = point_coord_pool.get() + (other_voxel.z_coords - other.point_coord_pool.get());
                }
            }
        }

        // ====================================================================
        // UTILITY FUNCTIONS
        // ====================================================================
        
        unsigned int next_power_of_two(unsigned int n) {
            if (n == 0) return 1;
            n--;
            n |= n >> 1;
            n |= n >> 2;
            n |= n >> 4;
            n |= n >> 8;
            n |= n >> 16;
            n++;
            return n;
        }

        [[nodiscard]] std::string get_robot_name() const {
            // Small epsilon for float comparison
            const float eps = 1e-3f;
            
            if (std::abs(max_query_radius - 0.07999999821186066) < eps &&
                std::abs(min_query_radius - 0.014999999664723873) < eps) {
                return "ur5";
            } else if (std::abs(max_query_radius - 0.07999999821186066) < eps && 
            std::abs(min_query_radius - 0.012000000104308128) < eps) {
                return "panda";
            } else if (std::abs(max_query_radius - 0.23999999463558197) < eps &&
                       std::abs(min_query_radius - 0.012000000104308128) < eps) {
                return "fetch";
            }
            return "UnknownRobot";
        }

        void record_mvt_state_to_file(const std::string& folder_path) const {
            std::string robot_name = get_robot_name();
            
            // Generate timestamp
            auto now = std::chrono::system_clock::now();
            auto in_time_t = std::chrono::system_clock::to_time_t(now);
            std::stringstream ss_filename;
            
            // Filename now includes the recognized robot name
            ss_filename << folder_path << "/" << robot_name << "_mvt_report_" 
                        << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S") << ".txt";
            
            std::ofstream out(ss_filename.str());
            if (!out.is_open()) {
                throw std::runtime_error("Could not open file: " + ss_filename.str());
            }
        
            out << "========================================================\n";
            out << "MVT REPORT FOR ROBOT: " << robot_name << "\n";
            out << "========================================================\n\n";
        
            // --- METADATA ---
            out << "[Metadata]\n";
            out << "Robot Type:                     " << robot_name << "\n";
            out << "Grid Width:                     " << grid_width << "x" << grid_width << "x" << grid_width << "\n";
            out << "Inverse Scale Factor:           " << inverse_scale_factor << "\n";
            out << "Point Radius:                   " << point_radius << "\n";
            out << "Max Query Radius:               " << max_query_radius << "\n";
            out << "Estimated Max_Point_Per_Voxel:  " << estimated_max_point_per_voxel << "\n";
            out << "Voxel Count (Used):   " << voxel_storage.size() << "\n\n";
        
            // --- MEMORY CONSUMPTION & UNUSED SPACE ---
            size_t point_used = point_coord_pool_used * sizeof(float);
            size_t point_total = point_coord_pool_size * sizeof(float);
            size_t hier_used = hierarchy_pool_used_bytes;
            size_t hier_total = hierarchy_pool_size_bytes;
            size_t voxel_meta = voxel_storage.size() * sizeof(Voxel);
        
            auto print_mem = [&](const std::string& label, size_t used, size_t total) {
                out << label << ":\n";
                out << "  Used:    " << std::fixed << std::setprecision(2) << used / 1024.0 << " KB\n";
                out << "  Unused:  " << (total - used) / 1024.0 << " KB\n";
                out << "  Efficiency: " << (total > 0 ? (used * 100.0 / total) : 0) << "%\n";
            };
        
            out << "[Memory Analysis]\n";
            print_mem("Point Coord Pool", point_used, point_total);
            print_mem("Hierarchy Pool  ", hier_used, hier_total);
            out << "Voxel Vector (Excluding pointed point coords):" << voxel_meta / 1024.0 << " KB\n";
            out << "Total Footprint:  " << (point_total + hier_total + voxel_meta) / 1024.0 << " KB\n\n";
        
            // --- HIERARCHY STATISTICS ---
            size_t x_entries = 0, y_entries = 0, z_entries = 0;
            const uint8_t* base = hierarchy_pool.get();
        
            for (uint16_t x = 0; x < grid_width; ++x) {
                if (x_level_table[x] != NULL_OFFSET) {
                    x_entries++;
                    const TableOffset* y_table = reinterpret_cast<const TableOffset*>(base + x_level_table[x]);
                    for (uint16_t y = 0; y < grid_width; ++y) {
                        if (y_table[y] != NULL_OFFSET) {
                            y_entries++;
                            const VoxelIndex* z_table = reinterpret_cast<const VoxelIndex*>(base + y_table[y]);
                            for (uint16_t z = 0; z < grid_width; ++z) {
                                if (z_table[z] != INVALID_VOXEL_INDEX) z_entries++;
                            }
                        }
                    }
                }
            }
        
            out << "[Structure Stats]\n";
            out << "Table Occupancy: X=" << x_entries << ", Y=" << y_entries << ", Z=" << z_entries << "\n";
    
            double x_null_rate = 100.0 * (1.0 - (static_cast<double>(x_entries) / grid_width));
            // Only calculate Y and Z if their parents exist to avoid Division by Zero
            double y_null_rate = (x_entries > 0) ? 
                100.0 * (1.0 - (static_cast<double>(y_entries) / (x_entries * grid_width))) : 100.0;
            double z_null_rate = (y_entries > 0) ? 
                100.0 * (1.0 - (static_cast<double>(z_entries) / (y_entries * grid_width))) : 100.0;

            out << "NULL_OFFSET Rate (Sparsity):\n"
                << "  X-Level: " << std::fixed << std::setprecision(2) << x_null_rate << "%\n"
                << "  Y-Level: " << y_null_rate << "%\n"
                << "  Z-Level: " << z_null_rate << "%\n\n";
                
            // --- PARTIAL VISUALIZATION ---
            out << "[Sampled Hierarchy Visualization (First 3 X-Slices)]\n";
            int count = 0;
            for (uint16_t x = 0; x < grid_width && count < 3; ++x) {
                if (x_level_table[x] == NULL_OFFSET) continue;
                out << "X[" << x << "]: ";
                const TableOffset* y_table = reinterpret_cast<const TableOffset*>(base + x_level_table[x]);
                for (uint16_t y = 0; y < grid_width; ++y) {
                    if (y_table[y] != NULL_OFFSET) out << "Y" << y << " ";
                }
                out << "\n";
                count++;
            }
        
            out.close();
        }

        struct QueryData {
            std::vector<std::vector<float>> x_coords;
            std::vector<std::vector<float>> y_coords;
            std::vector<std::vector<float>> z_coords;
            std::vector<std::vector<float>> radii;
        };
        
        // Parse query data from text file
        bool loadQueries(const std::string& filename, QueryData& queries) {
            std::ifstream file(filename);
            if (!file.is_open()) {
                std::cerr << "Error: Cannot open query file: " << filename << std::endl;
                return false;
            }
        
            queries.x_coords.clear();
            queries.y_coords.clear();
            queries.z_coords.clear();
            queries.radii.clear();
        
            std::string line;
            while (std::getline(file, line)) {
                // Parse line format: [ [x values] ] [ [y values] ] [ [z values] ] [ [radii] ]
                std::vector<std::vector<float>> line_data(4);
                
                std::istringstream iss(line);
                std::string token;
                int array_idx = 0;
                
                while (iss >> token && array_idx < 4) {
                    if (token == "[") {
                        // Skip opening bracket
                        continue;
                    } else if (token == "]") {
                        array_idx++;
                        continue;
                    } else if (token.front() == '[' && token.back() != ']') {
                        // Start of array, remove opening bracket
                        token = token.substr(1);
                    }
                    
                    // Clean up token (remove commas, brackets)
                    token.erase(std::remove(token.begin(), token.end(), ','), token.end());
                    if (token.back() == ']') {
                        token.pop_back();
                    }
                    
                    if (!token.empty()) {
                        try {
                            float value = std::stof(token);
                            line_data[array_idx].push_back(value);
                        } catch (const std::exception& e) {
                            // Skip invalid tokens
                        }
                    }
                }
                
                if (!line_data[0].empty()) {
                    queries.x_coords.push_back(line_data[0]);
                    queries.y_coords.push_back(line_data[1]);
                    queries.z_coords.push_back(line_data[2]);
                    queries.radii.push_back(line_data[3]);
                }
            }
            
            file.close();
            // std::cout << "Loaded " << queries.x_coords.size() << " query sets from " << filename << std::endl;
            return true;
        }
        
        void benchmark_collision_queries(const std::string& query_file) noexcept {
            // Load query data from file
            QueryData queries;
            if (!loadQueries(query_file, queries)) {
                std::cerr << "Failed to load queries from: " << query_file << std::endl;
                return;
            }
            
            if (queries.x_coords.empty()) {
                std::cerr << "No valid queries loaded from file" << std::endl;
                return;
            }
            
            // Validate that all coordinate arrays have the same size
            const size_t num_batches = queries.x_coords.size();
            std::cout << "num_batches = " << num_batches << std::endl;
            if (queries.y_coords.size() != num_batches || 
                queries.z_coords.size() != num_batches || 
                queries.radii.size() != num_batches) {
                std::cerr << "Error: Inconsistent batch sizes in query data" << std::endl;
                return;
            }
            
            // Flatten all queries into single vectors
            std::vector<float> all_x_coords;
            std::vector<float> all_y_coords;
            std::vector<float> all_z_coords;
            std::vector<float> all_radii;
            
            for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
                const auto& x_batch = queries.x_coords[batch_idx];
                const auto& y_batch = queries.y_coords[batch_idx];
                const auto& z_batch = queries.z_coords[batch_idx];
                const auto& r_batch = queries.radii[batch_idx];
                
                const size_t batch_size = x_batch.size();
                if (y_batch.size() != batch_size || 
                    z_batch.size() != batch_size || 
                    r_batch.size() != batch_size) {
                    std::cerr << "Warning: Inconsistent sizes in batch " << batch_idx << ", skipping..." << std::endl;
                    continue;
                }
                
                all_x_coords.insert(all_x_coords.end(), x_batch.begin(), x_batch.end());
                all_y_coords.insert(all_y_coords.end(), y_batch.begin(), y_batch.end());
                all_z_coords.insert(all_z_coords.end(), z_batch.begin(), z_batch.end());
                all_radii.insert(all_radii.end(), r_batch.begin(), r_batch.end());
            }
            
            const size_t total_individual_queries = all_x_coords.size();
            if (total_individual_queries == 0) {
                std::cerr << "No valid individual queries found" << std::endl;
                return;
            }
            
            std::cout << "Starting collision query benchmark with " << total_individual_queries << " individual queries..." << std::endl;
            
            constexpr size_t SIMD_WIDTH = 4;
            constexpr int NUM_TRIALS = 10;
            
            std::vector<double> simd_total_times(NUM_TRIALS, 0.0);
            std::vector<double> scalar_total_times(NUM_TRIALS, 0.0);
            
            size_t final_collisions_simd = 0;
            size_t final_collisions_scalar = 0;

            // 1. Benchmark: SIMD Version (10 Trials)
            for (int trial = 0; trial < NUM_TRIALS; ++trial) {
                size_t current_collisions = 0;
                
                auto start_time = std::chrono::steady_clock::now();
                
                for (size_t i = 0; i < total_individual_queries; i += SIMD_WIDTH) {
                    const size_t remaining = std::min(SIMD_WIDTH, total_individual_queries - i);
                    std::array<FVectorT, 3> centers;
                    FVectorT radii;
                    
                    for (size_t j = 0; j < remaining; ++j) {
                        centers[0][j] = all_x_coords[i + j];
                        centers[1][j] = all_y_coords[i + j];
                        centers[2][j] = all_z_coords[i + j];
                        radii[j] = all_radii[i + j];
                    }
                    
                    if (collides_simd(centers, radii)) {
                        current_collisions++;
                    }
                }
                
                auto end_time = std::chrono::steady_clock::now();
                simd_total_times[trial] = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
                final_collisions_simd = current_collisions;
            }

            // 2. Benchmark: Scalar Version (10 Trials)
            for (int trial = 0; trial < NUM_TRIALS; ++trial) {
                size_t current_collisions = 0;
                
                auto start_time = std::chrono::steady_clock::now();
                size_t batch_size = 4;
                for (size_t i = 0; i < total_individual_queries; i += batch_size) {
                    
                    // Process i to i + 3
                    for (size_t j = 0; j < batch_size && (i + j) < total_individual_queries; ++j) {
                        size_t idx = i + j;
                        
                        Point center{all_x_coords[idx], all_y_coords[idx], all_z_coords[idx]};
                        
                        if (collides(center, all_radii[idx])) {
                            current_collisions++; 
                            // Collide --> Skip this batch
                            break; 
                        }
                    }
                }
                auto end_time = std::chrono::steady_clock::now();
                scalar_total_times[trial] = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
                final_collisions_scalar = current_collisions;
            }

            // 3. Statistics
            double simd_sum = 0.0, scalar_sum = 0.0;
            for (int i = 0; i < NUM_TRIALS; ++i) {
                simd_sum += simd_total_times[i];
                scalar_sum += scalar_total_times[i];
            }
            
            double simd_avg_total = simd_sum / NUM_TRIALS;
            double simd_avg_per_query = simd_avg_total / total_individual_queries;
            
            double scalar_avg_total = scalar_sum / NUM_TRIALS;
            double scalar_avg_per_query = scalar_avg_total / total_individual_queries;

            // 4. Write to log
            std::ofstream log_file("scripts/log/cage_60_fetch_capt_q_mvt_rlt.txt", std::ios::app);
            if (log_file.is_open()) {
                log_file << "====================================================\n";
                log_file << "File: " << query_file << "\n";
                log_file << "Total individual queries: " << total_individual_queries << "\n";
                log_file << "Trials per method: " << NUM_TRIALS << "\n\n";

                // SIMD result
                log_file << "[SIMD Version] Collisions: " << final_collisions_simd << "\n";
                log_file << "10 Trials Total Time (ns): ";
                for (int i = 0; i < NUM_TRIALS; ++i) log_file << simd_total_times[i] << (i == NUM_TRIALS-1 ? "" : ", ");
                log_file << "\n";
                log_file << "Avg Total Time:     " << simd_avg_total << " ns\n";
                log_file << "Avg Time per Query: " << simd_avg_per_query << " ns\n\n";

                // Scalar result
                log_file << "[Scalar Version] Collisions: " << final_collisions_scalar << "\n";
                log_file << "10 Trials Total Time (ns): ";
                for (int i = 0; i < NUM_TRIALS; ++i) log_file << scalar_total_times[i] << (i == NUM_TRIALS-1 ? "" : ", ");
                log_file << "\n";
                log_file << "Avg Total Time:     " << scalar_avg_total << " ns\n";
                log_file << "Avg Time per Query: " << scalar_avg_per_query << " ns\n\n";

                // Summary
                log_file << "Speedup (Scalar Avg / SIMD Avg): " << (scalar_avg_per_query / simd_avg_per_query) << "x\n";
                log_file << "====================================================\n";
                
                log_file.close();
                std::cout << "Benchmark completed successfully. Log saved." << std::endl;
            } else {
                std::cerr << "Error: Could not open log file." << std::endl;
            }
        }

        void print_simd_args(const std::array<FVectorT, 3>& centers, 
                             FVectorT radii) const noexcept {
            constexpr size_t SIMD_WIDTH = FVectorT::num_scalars;
            alignas(32) float x[SIMD_WIDTH], y[SIMD_WIDTH], z[SIMD_WIDTH], r[SIMD_WIDTH];
        
            std::memcpy(x, &centers[0], sizeof(FVectorT));
            std::memcpy(y, &centers[1], sizeof(FVectorT));
            std::memcpy(z, &centers[2], sizeof(FVectorT));
            std::memcpy(r, &radii,      sizeof(FVectorT));
        
            auto print_lane = [](const float* data) {
                std::cout << "[ ";
                for (int i = 0; i < SIMD_WIDTH; ++i) {
                    std::cout << data[i] << (i == SIMD_WIDTH - 1 ? "" : " ");
                }
                std::cout << " ] ";
            };
        
            print_lane(x);
            print_lane(y);
            print_lane(z);
            print_lane(r);
            std::cout << std::endl;
        }

    };
}  // namespace vamp::collision