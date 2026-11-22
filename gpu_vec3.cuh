
#ifndef GPU_VEC3_CUH
#define GPU_VEC3_CUH

#include <cmath>

#include "gpu_camera.h"

__host__ __device__
inline GPUVec3 operator+(const GPUVec3& a, const GPUVec3& b) {
    return GPUVec3{
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    };
}

__host__ __device__
inline GPUVec3 operator-(const GPUVec3& a, const GPUVec3& b) {
    return GPUVec3{
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    };
}

__host__ __device__
inline GPUVec3 operator-(const GPUVec3& v) {
    return GPUVec3{-v.x, -v.y, -v.z};
}

__host__ __device__
inline GPUVec3 operator*(const GPUVec3& v, float scalar) {
    return GPUVec3{
        v.x * scalar,
        v.y * scalar,
        v.z * scalar
    };
}

__host__ __device__
inline GPUVec3 operator*(float scalar, const GPUVec3& v) {
    return v * scalar;
}

__host__ __device__
inline GPUVec3 operator/(const GPUVec3& v, float scalar) {
    return GPUVec3{
        v.x / scalar,
        v.y / scalar,
        v.z / scalar
    };
}

__host__ __device__
inline GPUVec3 operator*(const GPUVec3& a, const GPUVec3& b) {
    return GPUVec3{
        a.x * b.x,
        a.y * b.y,
        a.z * b.z
    };
}

__host__ __device__
inline float gpu_dot(const GPUVec3& a, const GPUVec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__
inline GPUVec3 gpu_cross(const GPUVec3& a, const GPUVec3& b) {
    return GPUVec3{
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

__host__ __device__
inline float gpu_length_squared(const GPUVec3& v) {
    return gpu_dot(v, v);
}

__host__ __device__
inline float gpu_length(const GPUVec3& v) {
    return sqrtf(gpu_length_squared(v));
}

__host__ __device__
inline GPUVec3 gpu_unit_vector(const GPUVec3& v) {
    float length = gpu_length(v);

    if (length <= 1e-20f) {
        return GPUVec3{0.0f, 0.0f, 0.0f};
    }

    return v / length;
}

__host__ __device__
inline GPUVec3 gpu_reflect(
    const GPUVec3& direction,
    const GPUVec3& normal
) {
    return direction - 2.0f * gpu_dot(direction, normal) * normal;
}

__host__ __device__
inline GPUVec3 gpu_refract(
    const GPUVec3& direction,
    const GPUVec3& normal,
    float eta
) {
    float cos_theta = fminf(
        gpu_dot(-direction, normal),
        1.0f
    );

    GPUVec3 perpendicular =
        eta * (direction + cos_theta * normal);

    GPUVec3 parallel =
        -sqrtf(fabsf(1.0f - gpu_length_squared(perpendicular))) *
        normal;

    return perpendicular + parallel;
}

#endif
