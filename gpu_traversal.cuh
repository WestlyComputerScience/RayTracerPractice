
#ifndef GPU_TRAVERSAL_CUH
#define GPU_TRAVERSAL_CUH

#include "gpu_scene.h"
#include "gpu_intersection.cuh"

struct GPUTraversalResult {
    float t;
    int primitive_index;
    int hit;
    int stack_overflow;
};

constexpr int GPU_BVH_STACK_SIZE = 64;

// Intersects a ray with a GPU quad.
__device__ bool gpu_quad_hit(
    const GPUPrimitive& primitive,
    const GPURay& ray,
    float ray_t_min,
    float ray_t_max,
    float& hit_t
) {
    const float qx = primitive.parameters[0];
    const float qy = primitive.parameters[1];
    const float qz = primitive.parameters[2];

    const float ux = primitive.parameters[3];
    const float uy = primitive.parameters[4];
    const float uz = primitive.parameters[5];

    const float vx = primitive.parameters[6];
    const float vy = primitive.parameters[7];
    const float vz = primitive.parameters[8];

    const float nx = primitive.parameters[9];
    const float ny = primitive.parameters[10];
    const float nz = primitive.parameters[11];

    const float D = primitive.parameters[12];

    float denom =
        nx * ray.direction_x +
        ny * ray.direction_y +
        nz * ray.direction_z;

    if (fabsf(denom) < 1e-8f) {
        return false;
    }

    float t = (
        D -
        nx * ray.origin_x -
        ny * ray.origin_y -
        nz * ray.origin_z
    ) / denom;

    if (t < ray_t_min || t > ray_t_max) {
        return false;
    }

    float px = ray.origin_x + t * ray.direction_x;
    float py = ray.origin_y + t * ray.direction_y;
    float pz = ray.origin_z + t * ray.direction_z;

    float hx = px - qx;
    float hy = py - qy;
    float hz = pz - qz;

    float cross_uv_x = uy * vz - uz * vy;
    float cross_uv_y = uz * vx - ux * vz;
    float cross_uv_z = ux * vy - uy * vx;

    float n_squared =
        cross_uv_x * cross_uv_x +
        cross_uv_y * cross_uv_y +
        cross_uv_z * cross_uv_z;

    if (n_squared == 0.0f) {
        return false;
    }

    float wx = cross_uv_x / n_squared;
    float wy = cross_uv_y / n_squared;
    float wz = cross_uv_z / n_squared;

    float cross_hv_x = hy * vz - hz * vy;
    float cross_hv_y = hz * vx - hx * vz;
    float cross_hv_z = hx * vy - hy * vx;

    float alpha =
        wx * cross_hv_x +
        wy * cross_hv_y +
        wz * cross_hv_z;

    float cross_uh_x = uy * hz - uz * hy;
    float cross_uh_y = uz * hx - ux * hz;
    float cross_uh_z = ux * hy - uy * hx;

    float beta =
        wx * cross_uh_x +
        wy * cross_uh_y +
        wz * cross_uh_z;

    if (alpha < 0.0f || alpha > 1.0f ||
        beta < 0.0f || beta > 1.0f) {
        return false;
    }

    hit_t = t;
    return true;
}

// Traverses the flattened GPU BVH and finds the closest quad hit.
__device__ GPUTraversalResult gpu_bvh_hit(
    const GPUBvhNode* nodes,
    const GPUPrimitive* primitives,
    int node_count,
    int primitive_count,
    const GPURay& ray,
    float ray_t_min,
    float ray_t_max
) {
    GPUTraversalResult result{};
    result.t = ray_t_max;
    result.primitive_index = -1;
    result.hit = 0;
    result.stack_overflow = 0;

    if (node_count == 0 || primitive_count == 0) {
        return result;
    }

    int stack[GPU_BVH_STACK_SIZE];
    int stack_size = 0;

    stack[stack_size++] = 0;

    while (stack_size > 0) {
        int node_index = stack[--stack_size];

        if (node_index < 0 || node_index >= node_count) {
            continue;
        }

        const GPUBvhNode& node = nodes[node_index];

        if (!gpu_aabb_hit(
            node,
            ray,
            ray_t_min,
            result.t
        )) {
            continue;
        }

        if (node.primitive_count > 0) {
            for (int i = 0; i < node.primitive_count; i++) {
                int primitive_index = node.primitive_start + i;

                if (primitive_index < 0 ||
                    primitive_index >= primitive_count) {
                    continue;
                }

                const GPUPrimitive& primitive =
                    primitives[primitive_index];

                if (primitive.type != GPUPrimitiveType::Quad) {
                    continue;
                }

                float hit_t;

                if (gpu_quad_hit(
                    primitive,
                    ray,
                    ray_t_min,
                    result.t,
                    hit_t
                )) {
                    result.t = hit_t;
                    result.primitive_index = primitive_index;
                    result.hit = 1;
                }
            }
        } else {
            int children[2] = {
                node.left_child,
                node.right_child
            };

            for (int i = 0; i < 2; i++) {
                int child_index = children[i];

                if (child_index < 0 || child_index >= node_count) {
                    continue;
                }

                if (gpu_aabb_hit(
                    nodes[child_index],
                    ray,
                    ray_t_min,
                    result.t
                )) {
                    if (stack_size >= GPU_BVH_STACK_SIZE) {
                        result.stack_overflow = 1;
                        return result;
                    }

                    stack[stack_size++] = child_index;
                }
            }
        }
    }

    return result;
}

#endif
