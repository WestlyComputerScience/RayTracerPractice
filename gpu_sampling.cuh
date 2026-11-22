
#ifndef GPU_SAMPLING_CUH
#define GPU_SAMPLING_CUH

#include "gpu_random.cuh"
#include "gpu_vec3.cuh"

// GPU equivalent of the CPU Onb class.
struct GPUOnb {
    GPUVec3 axis[3];

    __device__ explicit GPUOnb(const GPUVec3& normal) {
        axis[2] = gpu_unit_vector(normal);

        GPUVec3 a =
            (fabsf(axis[2].x) > 0.9f)
                ? GPUVec3{0.0f, 1.0f, 0.0f}
                : GPUVec3{1.0f, 0.0f, 0.0f};

        axis[1] = gpu_unit_vector(gpu_cross(axis[2], a));
        axis[0] = gpu_cross(axis[2], axis[1]);
    }

    __device__ const GPUVec3& u() const {
        return axis[0];
    }

    __device__ const GPUVec3& v() const {
        return axis[1];
    }

    __device__ const GPUVec3& w() const {
        return axis[2];
    }

    __device__ GPUVec3 transform(const GPUVec3& vector) const {
        return
            vector.x * axis[0] +
            vector.y * axis[1] +
            vector.z * axis[2];
    }
};

// Samples a random point inside the unit sphere.
__device__ inline GPUVec3 gpu_random_in_unit_sphere(
    unsigned int& seed
) {
    while (true) {
        GPUVec3 point{
            2.0f * gpu_random_float(seed) - 1.0f,
            2.0f * gpu_random_float(seed) - 1.0f,
            2.0f * gpu_random_float(seed) - 1.0f
        };

        if (gpu_length_squared(point) < 1.0f) {
            return point;
        }
    }
}

// Samples a random unit-length direction.
__device__ inline GPUVec3 gpu_random_unit_vector(
    unsigned int& seed
) {
    while (true) {
        GPUVec3 point = gpu_random_in_unit_sphere(seed);
        float length_squared = gpu_length_squared(point);

        if (length_squared > 1e-20f) {
            return point / sqrtf(length_squared);
        }
    }
}

// Samples a random point inside the hemisphere
// oriented in the direction of the supplied normal.
__device__ inline GPUVec3 gpu_random_in_hemisphere(
    const GPUVec3& normal,
    unsigned int& seed
) {
    GPUVec3 point = gpu_random_in_unit_sphere(seed);

    if (gpu_dot(point, normal) > 0.0f) {
        return point;
    }

    return -point;
}

// Generates a cosine-weighted direction in local coordinates.
__device__ inline GPUVec3 gpu_cosine_sample_hemisphere(
    unsigned int& seed
) {
    float r1 = gpu_random_float(seed);
    float r2 = gpu_random_float(seed);

    float phi = 2.0f * 3.14159265358979323846f * r1;

    float x = cosf(phi) * sqrtf(r2);
    float y = sinf(phi) * sqrtf(r2);
    float z = sqrtf(1.0f - r2);

    return GPUVec3{x, y, z};
}

// Generates a cosine-weighted direction around a surface normal.
__device__ inline GPUVec3 gpu_random_cosine_direction(
    const GPUVec3& normal,
    unsigned int& seed
) {
    GPUOnb basis(normal);

    return basis.transform(
        gpu_cosine_sample_hemisphere(seed)
    );
}

#endif
