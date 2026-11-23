
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "common_constants.h"
#include "vec3.h"
#include "interval.h"
#include "ray.h"
#include "pdf.h"
#include "gpu_material.cuh"

constexpr int TEST_COUNT = 6;
constexpr float TOLERANCE = 3e-5f;

__global__ void evaluate_gpu_vndf(
    const GPUVec3* views,
    const float* roughness_values,
    const float* r1_values,
    const float* r2_values,
    GPUVec3* results
) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= TEST_COUNT) {
        return;
    }

    results[index] = gpu_sample_ggx_vndf(
        views[index],
        roughness_values[index],
        r1_values[index],
        r2_values[index]
    );
}

int main() {
    // All view directions are normalized and above the local surface.
    Vec3 cpu_views[TEST_COUNT] = {
        unit_vector(Vec3(0.0, 0.0, 1.0)),
        unit_vector(Vec3(0.3, 0.9, 0.1)),
        unit_vector(Vec3(0.9, 0.2, -0.2)),
        unit_vector(Vec3(-0.7, 0.6, 0.3)),
        unit_vector(Vec3(0.1, 0.99, 0.05)),
        unit_vector(Vec3(0.8, 0.3, 0.4))
    };

    float roughness_values[TEST_COUNT] = {
        0.35f, 0.35f, 0.20f, 0.50f, 0.80f, 1.0f
    };

    // Identical deterministic random inputs for CPU and GPU.
    float r1_values[TEST_COUNT] = {
        0.01f, 0.25f, 0.50f, 0.75f, 0.99f, 0.40f
    };

    float r2_values[TEST_COUNT] = {
        0.10f, 0.30f, 0.50f, 0.70f, 0.90f, 0.25f
    };

    GPUVec3 gpu_views[TEST_COUNT];

    for (int i = 0; i < TEST_COUNT; i++) {
        gpu_views[i] = GPUVec3{
            static_cast<float>(cpu_views[i].x()),
            static_cast<float>(cpu_views[i].y()),
            static_cast<float>(cpu_views[i].z())
        };
    }

    GPUVec3* device_views = nullptr;
    GPUVec3* device_results = nullptr;
    float* device_roughness = nullptr;
    float* device_r1 = nullptr;
    float* device_r2 = nullptr;

    cudaMalloc(&device_views, sizeof(GPUVec3) * TEST_COUNT);
    cudaMalloc(&device_results, sizeof(GPUVec3) * TEST_COUNT);
    cudaMalloc(&device_roughness, sizeof(float) * TEST_COUNT);
    cudaMalloc(&device_r1, sizeof(float) * TEST_COUNT);
    cudaMalloc(&device_r2, sizeof(float) * TEST_COUNT);

    cudaMemcpy(
        device_views,
        gpu_views,
        sizeof(GPUVec3) * TEST_COUNT,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        device_roughness,
        roughness_values,
        sizeof(float) * TEST_COUNT,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        device_r1,
        r1_values,
        sizeof(float) * TEST_COUNT,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        device_r2,
        r2_values,
        sizeof(float) * TEST_COUNT,
        cudaMemcpyHostToDevice
    );

    evaluate_gpu_vndf<<<1, TEST_COUNT>>>(
        device_views,
        device_roughness,
        device_r1,
        device_r2,
        device_results
    );

    cudaError_t kernel_error = cudaGetLastError();
    cudaError_t sync_error = cudaDeviceSynchronize();

    if (kernel_error != cudaSuccess || sync_error != cudaSuccess) {
        std::cerr << "CUDA error: "
                  << cudaGetErrorString(
                      kernel_error != cudaSuccess
                          ? kernel_error
                          : sync_error
                  ) << "\n";

        cudaFree(device_views);
        cudaFree(device_results);
        cudaFree(device_roughness);
        cudaFree(device_r1);
        cudaFree(device_r2);

        return 1;
    }

    GPUVec3 gpu_results[TEST_COUNT];

    cudaMemcpy(
        gpu_results,
        device_results,
        sizeof(GPUVec3) * TEST_COUNT,
        cudaMemcpyDeviceToHost
    );

    bool all_passed = true;

    std::cout << "GGX VNDF CPU/GPU comparison:\n\n";

    for (int i = 0; i < TEST_COUNT; i++) {
        Vec3 cpu_result = sample_ggx_vndf(
            cpu_views[i],
            roughness_values[i],
            r1_values[i],
            r2_values[i]
        );

        GPUVec3 gpu_result = gpu_results[i];

        float dx = std::fabs(
            static_cast<float>(cpu_result.x()) - gpu_result.x
        );

        float dy = std::fabs(
            static_cast<float>(cpu_result.y()) - gpu_result.y
        );

        float dz = std::fabs(
            static_cast<float>(cpu_result.z()) - gpu_result.z
        );

        float cpu_length = static_cast<float>(cpu_result.length());

        float gpu_length = sqrtf(
            gpu_result.x * gpu_result.x +
            gpu_result.y * gpu_result.y +
            gpu_result.z * gpu_result.z
        );

        bool finite = std::isfinite(gpu_result.x) &&
                      std::isfinite(gpu_result.y) &&
                      std::isfinite(gpu_result.z);

        bool normalized = std::fabs(gpu_length - 1.0f) <= TOLERANCE;
        bool upper_hemisphere = gpu_result.z >= 0.0f;

        bool passed =
            finite &&
            normalized &&
            upper_hemisphere &&
            dx <= TOLERANCE &&
            dy <= TOLERANCE &&
            dz <= TOLERANCE;

        std::cout
            << "Sample " << i + 1 << ": "
            << (passed ? "PASS" : "FAIL") << "\n"
            << "  CPU: ("
            << cpu_result.x() << ", "
            << cpu_result.y() << ", "
            << cpu_result.z() << ")\n"
            << "  GPU: ("
            << gpu_result.x << ", "
            << gpu_result.y << ", "
            << gpu_result.z << ")\n"
            << "  Component differences: ("
            << dx << ", " << dy << ", " << dz << ")\n"
            << "  CPU length: " << cpu_length << "\n"
            << "  GPU length: " << gpu_length << "\n";

        if (!passed) {
            all_passed = false;
        }
    }

    cudaFree(device_views);
    cudaFree(device_results);
    cudaFree(device_roughness);
    cudaFree(device_r1);
    cudaFree(device_r2);

    std::cout << "\n";

    if (all_passed) {
        std::cout << "GPU GGX VNDF CPU comparison: PASS\n";
        return 0;
    }

    std::cout << "GPU GGX VNDF CPU comparison: FAIL\n";
    return 1;
}