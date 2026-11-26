
#ifndef GPU_TRAVERSAL_CUH
#define GPU_TRAVERSAL_CUH

#include "gpu_scene.h"
#include "gpu_intersection.cuh"

struct GPUTraversalResult {
    float t;

    int primitive_index;
    int hit;
    int stack_overflow;

    // World-space hit position.
    float position_x;
    float position_y;
    float position_z;

    // Surface normal.
    float normal_x;
    float normal_y;
    float normal_z;
};

constexpr int GPU_BVH_STACK_SIZE = 64;

// Intersects a ray with a GPU quad.
__device__ bool gpu_quad_hit(
    const GPUPrimitive& primitive,
    const GPURay& ray,
    float ray_t_min,
    float ray_t_max,
    float& hit_t,
    float& hit_x,
    float& hit_y,
    float& hit_z
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

    hit_x = px;
    hit_y = py;
    hit_z = pz;

    return true;
}

// Intersects a ray with a GPU sphere.
__device__ bool gpu_sphere_hit(
    const GPUPrimitive& primitive,
    const GPURay& ray,
    float ray_t_min,
    float ray_t_max,
    float& hit_t,
    float& hit_x,
    float& hit_y,
    float& hit_z,
    float& normal_x,
    float& normal_y,
    float& normal_z
) {
    const float center_x = primitive.parameters[0];
    const float center_y = primitive.parameters[1];
    const float center_z = primitive.parameters[2];

    const float radius = primitive.parameters[3];

    if (radius <= 0.0f) {
        return false;
    }

    const float oc_x = center_x - ray.origin_x;
    const float oc_y = center_y - ray.origin_y;
    const float oc_z = center_z - ray.origin_z;

    const float a =
        ray.direction_x * ray.direction_x +
        ray.direction_y * ray.direction_y +
        ray.direction_z * ray.direction_z;

    if (a <= 0.0f) {
        return false;
    }

    const float h =
        ray.direction_x * oc_x +
        ray.direction_y * oc_y +
        ray.direction_z * oc_z;

    const float c =
        oc_x * oc_x +
        oc_y * oc_y +
        oc_z * oc_z -
        radius * radius;

    const float discriminant = h * h - a * c;

    if (discriminant < 0.0f) {
        return false;
    }

    const float sqrt_discriminant = sqrtf(discriminant);

    // Try the closer intersection first.
    float t = (h - sqrt_discriminant) / a;

    // Match CPU Interval::surrounds(): endpoints are excluded.
    if (t <= ray_t_min || t >= ray_t_max) {
        // Try the farther intersection.
        t = (h + sqrt_discriminant) / a;

        if (t <= ray_t_min || t >= ray_t_max) {
            return false;
        }
    }

    const float px =
        ray.origin_x + t * ray.direction_x;

    const float py =
        ray.origin_y + t * ray.direction_y;

    const float pz =
        ray.origin_z + t * ray.direction_z;

    // Calculate the outward surface normal.
    float nx = (px - center_x) / radius;
    float ny = (py - center_y) / radius;
    float nz = (pz - center_z) / radius;

    // Match HitRecord::set_face_normal().
    const float direction_dot_normal =
        ray.direction_x * nx +
        ray.direction_y * ny +
        ray.direction_z * nz;

    if (direction_dot_normal >= 0.0f) {
        nx = -nx;
        ny = -ny;
        nz = -nz;
    }

    hit_t = t;

    hit_x = px;
    hit_y = py;
    hit_z = pz;

    normal_x = nx;
    normal_y = ny;
    normal_z = nz;

    return true;
}

// Tests a candidate Cone intersection root, including clipping
// and the oriented surface normal.
__device__ bool gpu_cone_test_root(
    const GPUPrimitive& primitive,
    const GPURay& ray,
    float ray_t_min,
    float ray_t_max,
    float t,
    float& hit_t,
    float& hit_x,
    float& hit_y,
    float& hit_z,
    float& normal_x,
    float& normal_y,
    float& normal_z
) {
    // Match CPU Interval::surrounds(): exclude both endpoints.
    if (t <= ray_t_min || t >= ray_t_max) {
        return false;
    }

    const float center_x = primitive.parameters[0];
    const float center_y = primitive.parameters[1];
    const float center_z = primitive.parameters[2];

    const float height = primitive.parameters[4];
    const float k = primitive.parameters[5];

    const float px =
        ray.origin_x + t * ray.direction_x;

    const float py =
        ray.origin_y + t * ray.direction_y;

    const float pz =
        ray.origin_z + t * ray.direction_z;

    // Cone clipping is performed in local coordinates.
    const float local_x = px - center_x;
    const float local_y = py - center_y;
    const float local_z = pz - center_z;

    if (local_y < 0.0f || local_y > height) {
        return false;
    }

    // Compute the outward normal.
    const float radial_length = sqrtf(
        local_x * local_x +
        local_z * local_z
    );

    float nx;
    float ny;
    float nz;

    // Match the CPU Cone apex normal.
    if (radial_length == 0.0f) {
        nx = 0.0f;
        ny = 1.0f;
        nz = 0.0f;
    } else {
        nx = local_x;
        ny = -k * radial_length;
        nz = local_z;

        const float normal_length = sqrtf(
            nx * nx +
            ny * ny +
            nz * nz
        );

        if (normal_length == 0.0f) {
            return false;
        }

        const float inverse_length = 1.0f / normal_length;

        nx *= inverse_length;
        ny *= inverse_length;
        nz *= inverse_length;
    }

    // Match HitRecord::set_face_normal().
    const float direction_dot_normal =
        ray.direction_x * nx +
        ray.direction_y * ny +
        ray.direction_z * nz;

    if (direction_dot_normal >= 0.0f) {
        nx = -nx;
        ny = -ny;
        nz = -nz;
    }

    hit_t = t;

    hit_x = px;
    hit_y = py;
    hit_z = pz;

    normal_x = nx;
    normal_y = ny;
    normal_z = nz;

    return true;
}

// Intersects a ray with a GPU Cone.
__device__ bool gpu_cone_hit(
    const GPUPrimitive& primitive,
    const GPURay& ray,
    float ray_t_min,
    float ray_t_max,
    float& hit_t,
    float& hit_x,
    float& hit_y,
    float& hit_z,
    float& normal_x,
    float& normal_y,
    float& normal_z
) {
    const float center_x = primitive.parameters[0];
    const float center_y = primitive.parameters[1];
    const float center_z = primitive.parameters[2];

    const float k = primitive.parameters[5];
    const float k_squared = k * k;

    // Ray origin relative to the Cone's center.
    const float ox = ray.origin_x - center_x;
    const float oy = ray.origin_y - center_y;
    const float oz = ray.origin_z - center_z;

    const float dx = ray.direction_x;
    const float dy = ray.direction_y;
    const float dz = ray.direction_z;

    // Solve:
    //
    // x^2 + z^2 - k^2*y^2 = 0
    //
    // after substituting the ray equation.
    const float a =
        dx * dx +
        dz * dz -
        k_squared * dy * dy;

    const float b = 2.0f * (
        ox * dx +
        oz * dz -
        k_squared * oy * dy
    );

    const float c =
        ox * ox +
        oz * oz -
        k_squared * oy * oy;

    // Handle the linear-degenerate case.
    if (a == 0.0f) {
        if (b == 0.0f) {
            return false;
        }

        const float t = -c / b;

        return gpu_cone_test_root(
            primitive,
            ray,
            ray_t_min,
            ray_t_max,
            t,
            hit_t,
            hit_x,
            hit_y,
            hit_z,
            normal_x,
            normal_y,
            normal_z
        );
    }

    const float discriminant = b * b - 4.0f * a * c;

    if (discriminant < 0.0f) {
        return false;
    }

    const float sqrt_discriminant = sqrtf(discriminant);

    // Try the nearer root first.
    const float first_root =
        (-b - sqrt_discriminant) / (2.0f * a);

    if (gpu_cone_test_root(
        primitive,
        ray,
        ray_t_min,
        ray_t_max,
        first_root,
        hit_t,
        hit_x,
        hit_y,
        hit_z,
        normal_x,
        normal_y,
        normal_z
    )) {
        return true;
    }

    // The nearer root may be outside the clipped Cone.
    // Try the farther root.
    const float second_root =
        (-b + sqrt_discriminant) / (2.0f * a);

    return gpu_cone_test_root(
        primitive,
        ray,
        ray_t_min,
        ray_t_max,
        second_root,
        hit_t,
        hit_x,
        hit_y,
        hit_z,
        normal_x,
        normal_y,
        normal_z
    );
}

__device__ bool gpu_quadric_test_root(
    const GPUPrimitive& primitive,
    const GPURay& ray,
    float ray_t_min,
    float ray_t_max,
    float t,
    float& hit_t,
    float& hit_x,
    float& hit_y,
    float& hit_z,
    float& normal_x,
    float& normal_y,
    float& normal_z
) {
    if (t <= ray_t_min || t >= ray_t_max) {
        return false;
    }

    const float center_x = primitive.parameters[0];
    const float center_y = primitive.parameters[1];
    const float center_z = primitive.parameters[2];

    const float px = ray.origin_x + t * ray.direction_x;
    const float py = ray.origin_y + t * ray.direction_y;
    const float pz = ray.origin_z + t * ray.direction_z;

    const float local_x = px - center_x;
    const float local_y = py - center_y;
    const float local_z = pz - center_z;

    float nx = 0.0f;
    float ny = 0.0f;
    float nz = 0.0f;

    if (primitive.type == GPUPrimitiveType::Cylinder) {
        const float height = primitive.parameters[4];

        if (local_y < 0.0f || local_y > height) {
            return false;
        }

        nx = local_x;
        ny = 0.0f;
        nz = local_z;
    }
    else if (primitive.type == GPUPrimitiveType::Hyperboloid) {
        const float height = primitive.parameters[4];
        const float flare = primitive.parameters[5];

        const float half_height = height / 2.0f;

        if (
            local_y < -half_height ||
            local_y > half_height
        ) {
            return false;
        }

        const float flare_squared = flare * flare;

        nx = local_x;
        ny = -flare_squared * local_y;
        nz = local_z;
    }
    else if (
        primitive.type ==
        GPUPrimitiveType::HyperbolicParaboloid
    ) {
        const float size = primitive.parameters[3];
        const float height = primitive.parameters[4];

        const float radius_squared =
            local_x * local_x +
            local_z * local_z;

        const float half_height = height / 2.0f;

        if (
            radius_squared > size * size ||
            local_y < -half_height ||
            local_y > half_height
        ) {
            return false;
        }

        nx = 2.0f * local_x;
        ny = -1.0f;
        nz = -2.0f * local_z;
    }
    else {
        return false;
    }

    const float normal_length = sqrtf(
        nx * nx +
        ny * ny +
        nz * nz
    );

    if (normal_length == 0.0f) {
        return false;
    }

    const float inverse_length = 1.0f / normal_length;

    nx *= inverse_length;
    ny *= inverse_length;
    nz *= inverse_length;

    // Match CPU HitRecord::set_face_normal().
    const float direction_dot_normal =
        ray.direction_x * nx +
        ray.direction_y * ny +
        ray.direction_z * nz;

    if (direction_dot_normal >= 0.0f) {
        nx = -nx;
        ny = -ny;
        nz = -nz;
    }

    hit_t = t;

    hit_x = px;
    hit_y = py;
    hit_z = pz;

    normal_x = nx;
    normal_y = ny;
    normal_z = nz;

    return true;
}


__device__ bool gpu_quadric_hit(
    const GPUPrimitive& primitive,
    const GPURay& ray,
    float ray_t_min,
    float ray_t_max,
    float& hit_t,
    float& hit_x,
    float& hit_y,
    float& hit_z,
    float& normal_x,
    float& normal_y,
    float& normal_z
) {
    const float center_x = primitive.parameters[0];
    const float center_y = primitive.parameters[1];
    const float center_z = primitive.parameters[2];

    const float ox = ray.origin_x - center_x;
    const float oy = ray.origin_y - center_y;
    const float oz = ray.origin_z - center_z;

    const float dx = ray.direction_x;
    const float dy = ray.direction_y;
    const float dz = ray.direction_z;

    float a;
    float b;
    float c;

    if (primitive.type == GPUPrimitiveType::Cylinder) {
        const float radius = primitive.parameters[3];

        a = dx * dx + dz * dz;

        b = 2.0f * (
            ox * dx +
            oz * dz
        );

        c =
            ox * ox +
            oz * oz -
            radius * radius;
    }
    else if (primitive.type == GPUPrimitiveType::Hyperboloid) {
        const float flare = primitive.parameters[5];
        const float flare_squared = flare * flare;

        a =
            dx * dx +
            dz * dz -
            flare_squared * dy * dy;

        b = 2.0f * (
            ox * dx +
            oz * dz -
            flare_squared * oy * dy
        );

        c =
            ox * ox +
            oz * oz -
            flare_squared * oy * oy -
            primitive.parameters[3] * primitive.parameters[3];
    }
    else if (
        primitive.type ==
        GPUPrimitiveType::HyperbolicParaboloid
    ) {
        a = dx * dx - dz * dz;

        b = 2.0f * (
            ox * dx -
            oz * dz
        ) - dy;

        c =
            ox * ox -
            oz * oz -
            oy;
    }
    else {
        return false;
    }

    // Handle the linear-degenerate case.
    if (a == 0.0f) {
        if (b == 0.0f) {
            return false;
        }

        const float t = -c / b;

        return gpu_quadric_test_root(
            primitive,
            ray,
            ray_t_min,
            ray_t_max,
            t,
            hit_t,
            hit_x,
            hit_y,
            hit_z,
            normal_x,
            normal_y,
            normal_z
        );
    }

    const float discriminant = b * b - 4.0f * a * c;

    if (discriminant < 0.0f) {
        return false;
    }

    const float sqrt_discriminant = sqrtf(discriminant);

    const float first_root =
        (-b - sqrt_discriminant) / (2.0f * a);

    if (gpu_quadric_test_root(
        primitive,
        ray,
        ray_t_min,
        ray_t_max,
        first_root,
        hit_t,
        hit_x,
        hit_y,
        hit_z,
        normal_x,
        normal_y,
        normal_z
    )) {
        return true;
    }

    const float second_root =
        (-b + sqrt_discriminant) / (2.0f * a);

    return gpu_quadric_test_root(
        primitive,
        ray,
        ray_t_min,
        ray_t_max,
        second_root,
        hit_t,
        hit_x,
        hit_y,
        hit_z,
        normal_x,
        normal_y,
        normal_z
    );
}

// Traverses the flattened GPU BVH and finds the closest primitive hit
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
        const int node_index = stack[--stack_size];

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
                const int primitive_index =
                    node.primitive_start + i;

                if (primitive_index < 0 ||
                    primitive_index >= primitive_count) {
                    continue;
                }

                const GPUPrimitive& primitive =
                    primitives[primitive_index];

                float hit_t;
                float hit_x;
                float hit_y;
                float hit_z;

                float hit_normal_x;
                float hit_normal_y;
                float hit_normal_z;

                bool primitive_hit = false;

                if (primitive.type == GPUPrimitiveType::Quad) {
                    primitive_hit = gpu_quad_hit(
                        primitive,
                        ray,
                        ray_t_min,
                        result.t,
                        hit_t,
                        hit_x,
                        hit_y,
                        hit_z
                    );

                    if (primitive_hit) {
                        hit_normal_x = primitive.parameters[9];
                        hit_normal_y = primitive.parameters[10];
                        hit_normal_z = primitive.parameters[11];
                    }
                }
                else if (primitive.type == GPUPrimitiveType::Sphere) {
                    primitive_hit = gpu_sphere_hit(
                        primitive,
                        ray,
                        ray_t_min,
                        result.t,
                        hit_t,
                        hit_x,
                        hit_y,
                        hit_z,
                        hit_normal_x,
                        hit_normal_y,
                        hit_normal_z
                    );
                }
                else if (primitive.type == GPUPrimitiveType::Cone) {
                    primitive_hit = gpu_cone_hit(
                        primitive,
                        ray,
                        ray_t_min,
                        result.t,
                        hit_t,
                        hit_x,
                        hit_y,
                        hit_z,
                        hit_normal_x,
                        hit_normal_y,
                        hit_normal_z
                    );
                }
                else if (
                    primitive.type == GPUPrimitiveType::Cylinder ||
                    primitive.type == GPUPrimitiveType::Hyperboloid ||
                    primitive.type == GPUPrimitiveType::HyperbolicParaboloid
                ) {
                    primitive_hit = gpu_quadric_hit(
                        primitive,
                        ray,
                        ray_t_min,
                        result.t,
                        hit_t,
                        hit_x,
                        hit_y,
                        hit_z,
                        hit_normal_x,
                        hit_normal_y,
                        hit_normal_z
                    );
                }

                if (primitive_hit) {
                    result.t = hit_t;
                    result.primitive_index = primitive_index;
                    result.hit = 1;

                    result.position_x = hit_x;
                    result.position_y = hit_y;
                    result.position_z = hit_z;

                    result.normal_x = hit_normal_x;
                    result.normal_y = hit_normal_y;
                    result.normal_z = hit_normal_z;
                }
            }
        }
        else {
            const int children[2] = {
                node.left_child,
                node.right_child
            };

            for (int i = 0; i < 2; i++) {
                const int child_index = children[i];

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