#ifndef GPU_PDF_CUH
#define GPU_PDF_CUH

#include "gpu_scene.h"
#include "gpu_intersection.cuh"
#include "gpu_random.cuh"

// GPU equivalent of CosinePdf.
struct GPUCosinePdf {
    float normal_x;
    float normal_y;
    float normal_z;
};

// GPU equivalent of HittablePdf.
struct GPUHittablePdf {
    float origin_x;
    float origin_y;
    float origin_z;
};

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