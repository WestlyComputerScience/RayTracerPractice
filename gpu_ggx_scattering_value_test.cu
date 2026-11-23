
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>

#include "gpu_material.cuh"

constexpr float PI = 3.14159265358979323846f;
constexpr float EPSILON = 2e-5f;

struct Vec3 {
    float x, y, z;
};

Vec3 normalize(Vec3 v) {
    float length = std::sqrt(
        v.x * v.x + v.y * v.y + v.z * v.z
    );

    return {
        v.x / length,
        v.y / length,
        v.z / length
    };
}

float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Independent CPU implementation of GGX distribution.
float cpu_distribution_ggx(
    Vec3 n,
    Vec3 h,
    float roughness
) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;

    float n_dot_h = std::max(dot(n, h), 0.0f);
    float n_dot_h2 = n_dot_h * n_dot_h;

    float denominator =
        n_dot_h2 * (alpha2 - 1.0f) + 1.0f;

    return alpha2 /
        (PI * denominator * denominator);
}

// Independent CPU implementation of Smith G1.
float cpu_geometry_smith_g1(
    float n_dot_v,
    float roughness
) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;

    float denominator = n_dot_v + std::sqrt(
        alpha2 +
        (1.0f - alpha2) * n_dot_v * n_dot_v
    );

    return (2.0f * n_dot_v) /
        std::max(denominator, 1e-8f);
}

// Independent CPU implementation of Cook-Torrance scattering.
Vec3 cpu_ggx_scattering_value(
    Vec3 normal,
    Vec3 view,
    Vec3 light,
    Vec3 albedo,
    float roughness,
    float metalness
) {
    float n_dot_v = std::max(dot(normal, view), 0.0f);
    float n_dot_l = std::max(dot(normal, light), 0.0f);

    if (n_dot_v <= 1e-8f || n_dot_l <= 0.0f) {
        return {0.0f, 0.0f, 0.0f};
    }

    Vec3 half_vector = normalize({
        view.x + light.x,
        view.y + light.y,
        view.z + light.z
    });

    float v_dot_h = std::max(
        dot(view, half_vector),
        0.0f
    );

    float fresnel_factor = std::pow(
        1.0f - v_dot_h,
        5.0f
    );

    Vec3 f0{
        0.04f * (1.0f - metalness) + albedo.x * metalness,
        0.04f * (1.0f - metalness) + albedo.y * metalness,
        0.04f * (1.0f - metalness) + albedo.z * metalness
    };

    Vec3 fresnel{
        f0.x + (1.0f - f0.x) * fresnel_factor,
        f0.y + (1.0f - f0.y) * fresnel_factor,
        f0.z + (1.0f - f0.z) * fresnel_factor
    };

    float D = cpu_distribution_ggx(
        normal,
        half_vector,
        roughness
    );

    float G_view = cpu_geometry_smith_g1(
        n_dot_v,
        roughness
    );

    float G_light = cpu_geometry_smith_g1(
        n_dot_l,
        roughness
    );

    float G = G_view * G_light;

    float factor = D * G / (4.0f * n_dot_v);

    return {
        fresnel.x * factor,
        fresnel.y * factor,
        fresnel.z * factor
    };
}

__global__ void test_gpu_ggx_scattering(
    float* results
) {
    GPUMaterial material{};

    material.type = GPUMaterialType::GGX;

    material.albedo_r = 0.8f;
    material.albedo_g = 0.4f;
    material.albedo_b = 0.2f;

    material.roughness = 0.35f;
    material.metalness = 0.6f;

    GPUVec3 normal{0.0f, 1.0f, 0.0f};

    GPUVec3 view = gpu_unit_vector(
        GPUVec3{0.3f, 0.9f, -0.1f}
    );

    GPUVec3 light = gpu_unit_vector(
        GPUVec3{0.2f, 0.95f, 0.25f}
    );

    // Incoming ray travels opposite the view direction.
    GPURay incoming_ray{
        0.0f, 1.0f, 0.0f,
        -view.x, -view.y, -view.z
    };

    GPUTraversalResult hit{};

    hit.hit = 1;
    hit.position_x = 0.0f;
    hit.position_y = 0.0f;
    hit.position_z = 0.0f;

    hit.normal_x = normal.x;
    hit.normal_y = normal.y;
    hit.normal_z = normal.z;

    unsigned int seed = 12345u;

    GPUScatterRecord scatter{};

    bool scattered = gpu_material_scatter(
        material,
        incoming_ray,
        hit,
        seed,
        scatter
    );

    GPUVec3 scattering_value =
        gpu_material_scattering_value(
            material,
            scatter,
            incoming_ray,
            hit,
            light
        );

    results[0] = scattered ? 1.0f : 0.0f;

    results[1] = scatter.attenuation.x;
    results[2] = scatter.attenuation.y;
    results[3] = scatter.attenuation.z;

    results[4] = scattering_value.x;
    results[5] = scattering_value.y;
    results[6] = scattering_value.z;
}

bool approximately_equal(float a, float b) {
    return std::fabs(a - b) <= EPSILON;
}

int main() {
    float* device_results = nullptr;
    float results[7] = {};

    cudaError_t error = cudaMalloc(
        &device_results,
        sizeof(results)
    );

    if (error != cudaSuccess) {
        std::cerr << "cudaMalloc failed: "
                  << cudaGetErrorString(error) << '\n';
        return 1;
    }

    test_gpu_ggx_scattering<<<1, 1>>>(device_results);

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        std::cerr << "Kernel failed: "
                  << cudaGetErrorString(error) << '\n';

        cudaFree(device_results);
        return 1;
    }

    error = cudaMemcpy(
        results,
        device_results,
        sizeof(results),
        cudaMemcpyDeviceToHost
    );

    cudaFree(device_results);

    if (error != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: "
                  << cudaGetErrorString(error) << '\n';
        return 1;
    }

    Vec3 normal{0.0f, 1.0f, 0.0f};

    Vec3 view = normalize({
        0.3f, 0.9f, -0.1f
    });

    Vec3 light = normalize({
        0.2f, 0.95f, 0.25f
    });

    Vec3 albedo{0.8f, 0.4f, 0.2f};

    Vec3 expected_f0{
        0.04f * 0.4f + 0.8f * 0.6f,
        0.04f * 0.4f + 0.4f * 0.6f,
        0.04f * 0.4f + 0.2f * 0.6f
    };

    Vec3 expected = cpu_ggx_scattering_value(
        normal,
        view,
        light,
        albedo,
        0.35f,
        0.6f
    );

    bool scatter_pass =
        approximately_equal(results[0], 1.0f);

    bool f0_pass =
        approximately_equal(results[1], expected_f0.x) &&
        approximately_equal(results[2], expected_f0.y) &&
        approximately_equal(results[3], expected_f0.z);

    bool value_pass =
        approximately_equal(results[4], expected.x) &&
        approximately_equal(results[5], expected.y) &&
        approximately_equal(results[6], expected.z);

    std::cout << "GGX scatter initialization: "
              << (scatter_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "GGX F0 calculation: "
              << (f0_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "GGX Cook-Torrance scattering value: "
              << (value_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "\nGPU scattering value:\n"
              << results[4] << ", "
              << results[5] << ", "
              << results[6] << '\n';

    std::cout << "CPU reference value:\n"
              << expected.x << ", "
              << expected.y << ", "
              << expected.z << '\n';

    bool all_passed =
        scatter_pass && f0_pass && value_pass;

    std::cout << "\nGPU GGX scattering tests: "
              << (all_passed ? "PASS" : "FAIL") << '\n';

    return all_passed ? 0 : 1;
}