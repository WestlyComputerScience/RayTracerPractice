#ifndef GPU_INTERSECTION_CUH
#define GPU_INTERSECTION_CUH

#include "gpu_scene.h"

struct GPURay {
    float origin_x, origin_y, origin_z;
    float direction_x, direction_y, direction_z;
};

// Returns whether a ray intersects a GPU bounding box.
__host__ __device__
inline bool gpu_aabb_hit(
    const GPUBvhNode& box,
    const GPURay& ray,
    float ray_t_min,
    float ray_t_max
) {
    const float origins[3] = {
        ray.origin_x,
        ray.origin_y,
        ray.origin_z
    };

    const float directions[3] = {
        ray.direction_x,
        ray.direction_y,
        ray.direction_z
    };

    const float box_min[3] = {
        box.min_x,
        box.min_y,
        box.min_z
    };

    const float box_max[3] = {
        box.max_x,
        box.max_y,
        box.max_z
    };

    for (int axis = 0; axis < 3; axis++) {
        // Handle parallel rays explicitly.
        if (directions[axis] == 0.0f) {
            if (origins[axis] < box_min[axis] ||
                origins[axis] > box_max[axis]) {
                return false;
            }

            continue;
        }

        const float inverse_direction =
            1.0f / directions[axis];

        float t0 =
            (box_min[axis] - origins[axis]) *
            inverse_direction;

        float t1 =
            (box_max[axis] - origins[axis]) *
            inverse_direction;

        if (t0 < t1) {
            if (t0 > ray_t_min) ray_t_min = t0;
            if (t1 < ray_t_max) ray_t_max = t1;
        } else {
            if (t1 > ray_t_min) ray_t_min = t1;
            if (t0 < ray_t_max) ray_t_max = t0;
        }

        // Equality is valid for zero-thickness bounding boxes.
        if (ray_t_max < ray_t_min) {
            return false;
        }
    }

    return true;
}

#endif