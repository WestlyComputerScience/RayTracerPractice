
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "gpu_material.cuh"

constexpr float EPSILON = 1e-5f;

__device__ void run_refraction_test(
    bool entering,
    float* results
) {
    GPUMaterial material{};

    material.type = GPUMaterialType::Dielectric;
    material.refractive_index = 1.5f;

    constexpr float sin_theta = 0.5f;
    constexpr float cos_theta = 0.86602540378f;

    GPURay ray{};

    if (entering) {
        // Air -> glass.
        ray = GPURay{
            0.0f, 1.0f, 0.0f,
            sin_theta, -cos_theta, 0.0f
        };
    } else {
        // Glass -> air.
        ray = GPURay{
            0.0f, -1.0f, 0.0f,
            sin_theta, cos_theta, 0.0f
        };
    }

    GPUTraversalResult hit{};

    hit.hit = 1;
    hit.position_y = 0.0f;
    hit.normal_y = 1.0f;

    unsigned int seed = 12345u;

    GPUScatterRecord scatter{};

    bool scattered = gpu_material_scatter(
        material,
        ray,
        hit,
        seed,
        scatter
    );

    results[0] = scattered ? 1.0f : 0.0f;
    results[1] = scatter.skip_pdf ? 1.0f : 0.0f;

    results[2] = scatter.skip_pdf_ray.direction_x;
    results[3] = scatter.skip_pdf_ray.direction_y;
    results[4] = scatter.skip_pdf_ray.direction_z;
}

__global__ void test_gpu_refraction(float* results) {
    run_refraction_test(true, &results[0]);
    run_refraction_test(false, &results[5]);
}

bool approximately_equal(float a, float b) {
    return std::fabs(a - b) <= EPSILON;
}

bool test_direction(
    const float* result,
    float expected_x,
    float expected_y
) {
    if (!approximately_equal(result[0], 1.0f)) {
        return false;
    }

    if (!approximately_equal(result[1], 1.0f)) {
        return false;
    }

    float x = result[2];
    float y = result[3];
    float z = result[4];

    float length = std::sqrt(x * x + y * y + z * z);

    return
        approximately_equal(x, expected_x) &&
        approximately_equal(y, expected_y) &&
        approximately_equal(z, 0.0f) &&
        approximately_equal(length, 1.0f);
}

int main() {
    float* device_results = nullptr;
    float results[10] = {};

    cudaError_t error = cudaMalloc(
        &device_results,
        sizeof(results)
    );

    if (error != cudaSuccess) {
        std::cerr << "cudaMalloc failed: "
                  << cudaGetErrorString(error) << '\n';
        return 1;
    }

    test_gpu_refraction<<<1, 1>>>(device_results);

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

    // Snell's law:
    // n1 * sin(theta1) = n2 * sin(theta2)

    // Air -> glass:
    // sin(theta2) = (1.0 / 1.5) * 0.5 = 1/3.
    // cos(theta2) = sqrt(8/9).
    const float entering_x = 1.0f / 3.0f;
    const float entering_y = -0.94280904158f;

    // Glass -> air:
    // sin(theta2) = (1.5 / 1.0) * 0.5 = 0.75.
    // cos(theta2) = sqrt(7/16).
    const float exiting_x = 0.75f;
    const float exiting_y = 0.66143782777f;

    bool entering_pass = test_direction(
        &results[0],
        entering_x,
        entering_y
    );

    bool exiting_pass = test_direction(
        &results[5],
        exiting_x,
        exiting_y
    );

    std::cout << "Air-to-glass refraction: "
              << (entering_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "Glass-to-air refraction: "
              << (exiting_pass ? "PASS" : "FAIL") << '\n';

    if (!entering_pass) {
        std::cout << "  Actual entering direction: ("
                  << results[2] << ", "
                  << results[3] << ", "
                  << results[4] << ")\n";
    }

    if (!exiting_pass) {
        std::cout << "  Actual exiting direction: ("
                  << results[7] << ", "
                  << results[8] << ", "
                  << results[9] << ")\n";
    }

    bool all_passed = entering_pass && exiting_pass;

    std::cout << "\nGPU dielectric refraction tests: "
              << (all_passed ? "PASS" : "FAIL") << '\n';

    return all_passed ? 0 : 1;
}