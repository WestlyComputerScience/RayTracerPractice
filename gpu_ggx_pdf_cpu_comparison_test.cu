
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "common_constants.h"
#include "vec3.h"
#include "interval.h"
#include "ray.h"
#include "pdf.h"
#include "gpu_material.cuh"

constexpr int DIRECTION_COUNT = 5;
constexpr float TOLERANCE = 3e-5f;

__global__ void evaluate_gpu_ggx_pdf(
    GPUVec3 normal,
    GPUVec3 view,
    float roughness,
    const GPUVec3* directions,
    float* results
) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= DIRECTION_COUNT) {
        return;
    }

    // Construct the GGX PDF on the GPU.
    GPUGGXPdf pdf = gpu_make_ggx_pdf(
        normal,
        view,
        roughness
    );

    GPUVec3 direction = directions[index];

    results[index] = gpu_ggx_pdf_value(
        pdf,
        direction.x,
        direction.y,
        direction.z
    );
}

int main() {
    // Identical inputs for CPU and GPU.
    Vec3 normal(0, 1, 0);
    Vec3 view = unit_vector(Vec3(0.3, 0.9, -0.1));
    real roughness = 0.35;

    GGXPdf cpu_pdf(normal, view, roughness);

    GPUVec3 gpu_normal{0.0f, 1.0f, 0.0f};

    GPUVec3 gpu_view{
        static_cast<float>(view.x()),
        static_cast<float>(view.y()),
        static_cast<float>(view.z())
    };

    // The same directions are evaluated by both implementations.
    GPUVec3 gpu_directions[DIRECTION_COUNT] = {
        { 0.2f,  0.95f,  0.25f},
        { 0.6f,  0.8f,   0.0f},
        {-0.3f,  0.7f,   0.4f},
        { 0.2f, -0.9f,   0.1f},
        { 0.0f,  0.0f,   0.0f}
    };

    Vec3 cpu_directions[DIRECTION_COUNT] = {
        Vec3( 0.2,  0.95,  0.25),
        Vec3( 0.6,  0.8,   0.0),
        Vec3(-0.3,  0.7,   0.4),
        Vec3( 0.2, -0.9,   0.1),
        Vec3( 0.0,  0.0,   0.0)
    };

    float* device_results = nullptr;
    GPUVec3* device_directions = nullptr;

    cudaMalloc(&device_results, sizeof(float) * DIRECTION_COUNT);
    cudaMalloc(&device_directions, sizeof(GPUVec3) * DIRECTION_COUNT);

    cudaMemcpy(
        device_directions,
        gpu_directions,
        sizeof(GPUVec3) * DIRECTION_COUNT,
        cudaMemcpyHostToDevice
    );

    evaluate_gpu_ggx_pdf<<<1, DIRECTION_COUNT>>>(
        gpu_normal,
        gpu_view,
        static_cast<float>(roughness),
        device_directions,
        device_results
    );

    cudaDeviceSynchronize();

    float gpu_results[DIRECTION_COUNT];

    cudaMemcpy(
        gpu_results,
        device_results,
        sizeof(float) * DIRECTION_COUNT,
        cudaMemcpyDeviceToHost
    );

    bool all_passed = true;

    std::cout << "GGX PDF CPU/GPU comparison:\n\n";

    for (int i = 0; i < DIRECTION_COUNT; i++) {
        real cpu_value = cpu_pdf.value(cpu_directions[i]);
        float gpu_value = gpu_results[i];

        bool cpu_finite = std::isfinite(static_cast<double>(cpu_value));
        bool passed = false;

        float difference = 0.0f;

        if (!cpu_finite) {
            // The CPU implementation produces NaN for a zero-length direction.
            // The GPU implementation should safely return zero.
            passed = std::isfinite(gpu_value) && gpu_value == 0.0f;

            std::cout
                << "Direction " << i + 1 << ": "
                << (passed ? "PASS" : "FAIL") << "\n"
                << "  CPU:        non-finite\n"
                << "  GPU:        " << gpu_value << "\n"
                << "  Expected GPU behavior: zero\n";
        }
        else {
            difference = std::fabs(
                static_cast<float>(cpu_value) - gpu_value
            );

            passed = std::isfinite(gpu_value) &&
                    difference <= TOLERANCE;

            std::cout
                << "Direction " << i + 1 << ": "
                << (passed ? "PASS" : "FAIL") << "\n"
                << "  CPU:        " << cpu_value << "\n"
                << "  GPU:        " << gpu_value << "\n"
                << "  Difference: " << difference << "\n";
        }

        if (!passed) {
            all_passed = false;
        }
    }

    cudaFree(device_results);
    cudaFree(device_directions);

    std::cout << "\n";

    if (all_passed) {
        std::cout << "GPU GGX PDF CPU comparison: PASS\n";
        return 0;
    }

    std::cout << "GPU GGX PDF CPU comparison: FAIL\n";
    return 1;
}