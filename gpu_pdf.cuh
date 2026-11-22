#ifndef GPU_PDF_CUH
#define GPU_PDF_CUH

#include "gpu_scene.h"
#include "gpu_intersection.cuh"
#include "gpu_random.cuh"
#include "gpu_sampling.cuh"

// GPU equivalent of CosinePdf.
struct GPUCosinePdf {
    float normal_x;
    float normal_y;
    float normal_z;
};

struct GPUSpherePdf {
};

__device__ inline float gpu_sphere_pdf_value() {
    return 1.0f / (4.0f * 3.14159265358979323846f);
}

__device__ inline void gpu_sphere_pdf_generate(
    unsigned int& seed,
    float& direction_x,
    float& direction_y,
    float& direction_z
) {
    GPUVec3 direction = gpu_random_unit_vector(seed);

    direction_x = direction.x;
    direction_y = direction.y;
    direction_z = direction.z;
}

// GPU equivalent of HittablePdf.
struct GPUHittablePdf {
    float origin_x;
    float origin_y;
    float origin_z;
};

__device__ inline float gpu_quad_pdf_value(
    const GPUPrimitive& primitive,
    const GPUHittablePdf& pdf,
    float direction_x,
    float direction_y,
    float direction_z
) {
    constexpr float epsilon = 1e-8f;

    // Read the quad's stored normal and plane constant.
    float normal_x = primitive.parameters[9];
    float normal_y = primitive.parameters[10];
    float normal_z = primitive.parameters[11];

    float D = primitive.parameters[12];
    float area = primitive.parameters[13];

    if (area < epsilon) {
        return 0.0f;
    }

    // Calculate the ray-plane denominator.
    float denominator =
        normal_x * direction_x +
        normal_y * direction_y +
        normal_z * direction_z;

    if (fabsf(denominator) < epsilon) {
        return 0.0f;
    }

    // Calculate the intersection distance.
    float origin_dot_normal =
        normal_x * pdf.origin_x +
        normal_y * pdf.origin_y +
        normal_z * pdf.origin_z;

    float t = (D - origin_dot_normal) / denominator;

    if (t <= 0.001f) {
        return 0.0f;
    }

    // Calculate the intersection point.
    float hit_x = pdf.origin_x + t * direction_x;
    float hit_y = pdf.origin_y + t * direction_y;
    float hit_z = pdf.origin_z + t * direction_z;

    // Calculate the planar coordinates.
    float q_x = primitive.parameters[0];
    float q_y = primitive.parameters[1];
    float q_z = primitive.parameters[2];

    float u_x = primitive.parameters[3];
    float u_y = primitive.parameters[4];
    float u_z = primitive.parameters[5];

    float v_x = primitive.parameters[6];
    float v_y = primitive.parameters[7];
    float v_z = primitive.parameters[8];

    float planar_x = hit_x - q_x;
    float planar_y = hit_y - q_y;
    float planar_z = hit_z - q_z;

    // w = cross(u, v) / |cross(u, v)|^2.
    float cross_x = u_y * v_z - u_z * v_y;
    float cross_y = u_z * v_x - u_x * v_z;
    float cross_z = u_x * v_y - u_y * v_x;

    float cross_length_squared =
        cross_x * cross_x +
        cross_y * cross_y +
        cross_z * cross_z;

    if (cross_length_squared < epsilon) {
        return 0.0f;
    }

    float w_x = cross_x / cross_length_squared;
    float w_y = cross_y / cross_length_squared;
    float w_z = cross_z / cross_length_squared;

    // alpha = dot(w, cross(planar_hit, v)).
    float alpha_cross_x =
        planar_y * v_z - planar_z * v_y;

    float alpha_cross_y =
        planar_z * v_x - planar_x * v_z;

    float alpha_cross_z =
        planar_x * v_y - planar_y * v_x;

    float alpha =
        w_x * alpha_cross_x +
        w_y * alpha_cross_y +
        w_z * alpha_cross_z;

    // beta = dot(w, cross(u, planar_hit)).
    float beta_cross_x =
        u_y * planar_z - u_z * planar_y;

    float beta_cross_y =
        u_z * planar_x - u_x * planar_z;

    float beta_cross_z =
        u_x * planar_y - u_y * planar_x;

    float beta =
        w_x * beta_cross_x +
        w_y * beta_cross_y +
        w_z * beta_cross_z;

    // The intersection must lie inside the parallelogram.
    if (
        alpha < 0.0f || alpha > 1.0f ||
        beta < 0.0f || beta > 1.0f
    ) {
        return 0.0f;
    }

    // Match Quad::pdf_value().
    float distance_squared =
        t * t * (
            direction_x * direction_x +
            direction_y * direction_y +
            direction_z * direction_z
        );

    float direction_length = sqrtf(
        direction_x * direction_x +
        direction_y * direction_y +
        direction_z * direction_z
    );

    if (direction_length < epsilon) {
        return 0.0f;
    }

    float cosine = fabsf(denominator / direction_length);

    if (cosine < epsilon) {
        return 0.0f;
    }

    return distance_squared / (cosine * area);
}

__device__ inline void gpu_quad_pdf_generate(
    const GPUPrimitive& primitive,
    const GPUHittablePdf& pdf,
    unsigned int& seed,
    float& direction_x,
    float& direction_y,
    float& direction_z
) {
    // Sample a random point on the quad.
    float r1 = gpu_random_float(seed);
    float r2 = gpu_random_float(seed);

    float point_x =
        primitive.parameters[0] +
        r1 * primitive.parameters[3] +
        r2 * primitive.parameters[6];

    float point_y =
        primitive.parameters[1] +
        r1 * primitive.parameters[4] +
        r2 * primitive.parameters[7];

    float point_z =
        primitive.parameters[2] +
        r1 * primitive.parameters[5] +
        r2 * primitive.parameters[8];

    // Return the direction from the sampling origin.
    direction_x = point_x - pdf.origin_x;
    direction_y = point_y - pdf.origin_y;
    direction_z = point_z - pdf.origin_z;
}


__device__ inline float gpu_hittable_pdf_value(
    const GPUPrimitive* primitives,
    const int* light_indices,
    int light_count,
    const GPUHittablePdf& pdf,
    float direction_x,
    float direction_y,
    float direction_z
) {
    if (
        light_count <= 0 ||
        primitives == nullptr ||
        light_indices == nullptr
    ) {
        return 0.0f;
    }

    float total_pdf = 0.0f;

    for (int i = 0; i < light_count; i++) {
        int primitive_index = light_indices[i];

        const GPUPrimitive& primitive =
            primitives[primitive_index];

        if (primitive.type != GPUPrimitiveType::Quad) {
            continue;
        }

        total_pdf += gpu_quad_pdf_value(
            primitive,
            pdf,
            direction_x,
            direction_y,
            direction_z
        );
    }

    return total_pdf / static_cast<float>(light_count);
}

__device__ inline bool gpu_hittable_pdf_generate(
    const GPUPrimitive* primitives,
    const int* light_indices,
    int light_count,
    const GPUHittablePdf& pdf,
    unsigned int& seed,
    float& direction_x,
    float& direction_y,
    float& direction_z
) {
    if (
        light_count <= 0 ||
        primitives == nullptr ||
        light_indices == nullptr
    ) {
        direction_x = 0.0f;
        direction_y = 0.0f;
        direction_z = 0.0f;
        return false;
    }

    // Uniformly select one light.
    int selected_light = static_cast<int>(
        gpu_random_float(seed) * light_count
    );

    int primitive_index = light_indices[selected_light];

    const GPUPrimitive& primitive =
        primitives[primitive_index];

    if (primitive.type != GPUPrimitiveType::Quad) {
        direction_x = 0.0f;
        direction_y = 0.0f;
        direction_z = 0.0f;
        return false;
    }

    gpu_quad_pdf_generate(
        primitive,
        pdf,
        seed,
        direction_x,
        direction_y,
        direction_z
    );

    return true;
}

__device__ inline float gpu_cosine_pdf_value(
    const GPUCosinePdf& pdf,
    float direction_x,
    float direction_y,
    float direction_z
) {
    float length = sqrtf(
        direction_x * direction_x +
        direction_y * direction_y +
        direction_z * direction_z
    );

    if (length <= 1e-8f) {
        return 0.0f;
    }

    float unit_x = direction_x / length;
    float unit_y = direction_y / length;
    float unit_z = direction_z / length;

    float cosine_theta =
        unit_x * pdf.normal_x +
        unit_y * pdf.normal_y +
        unit_z * pdf.normal_z;

    return fmaxf(0.0f, cosine_theta / 3.14159265358979323846f);
}

__device__ inline void gpu_cosine_pdf_generate(
    const GPUCosinePdf& pdf,
    unsigned int& seed,
    float& direction_x,
    float& direction_y,
    float& direction_z
) {
    constexpr float pi = 3.14159265358979323846f;

    // Generate two uniform random values in [0, 1).
    float r1 = gpu_random_float(seed);
    float r2 = gpu_random_float(seed);

    // Match CPU random_cosine_direction().
    float phi = 2.0f * pi * r1;

    float x = cosf(phi) * sqrtf(r2);
    float y = sinf(phi) * sqrtf(r2);
    float z = sqrtf(1.0f - r2);

    // Reconstruct the CPU Onb coordinate system.
    float w_x = pdf.normal_x;
    float w_y = pdf.normal_y;
    float w_z = pdf.normal_z;

    float a_x = fabsf(w_x) > 0.9f ? 0.0f : 1.0f;
    float a_y = fabsf(w_x) > 0.9f ? 1.0f : 0.0f;
    float a_z = 0.0f;

    // v = unit_vector(cross(w, a))
    float v_x = w_y * a_z - w_z * a_y;
    float v_y = w_z * a_x - w_x * a_z;
    float v_z = w_x * a_y - w_y * a_x;

    float v_length = sqrtf(
        v_x * v_x + v_y * v_y + v_z * v_z
    );

    v_x /= v_length;
    v_y /= v_length;
    v_z /= v_length;

    // u = cross(w, v)
    float u_x = w_y * v_z - w_z * v_y;
    float u_y = w_z * v_x - w_x * v_z;
    float u_z = w_x * v_y - w_y * v_x;

    // Transform the local cosine-weighted direction.
    direction_x = x * u_x + y * v_x + z * w_x;
    direction_y = x * u_y + y * v_y + z * w_y;
    direction_z = x * u_z + y * v_z + z * w_z;
}

#endif