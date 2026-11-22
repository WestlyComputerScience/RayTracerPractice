#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "gpu_sampling.cuh"

constexpr int SAMPLE_COUNT = 4096;

__global__ void test_gpu_sampling(
    GPUVec3* results,
    int sample_count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= sample_count) {
        return;
    }

    unsigned int seed =
        static_cast<unsigned int>(index) + 1u;

    GPUVec3 normal{0.0f, 1.0f, 0.0f};

    results[index] =
        gpu_random_in_unit_sphere(seed);

    results[sample_count + index] =
        gpu_random_unit_vector(seed);

    results[2 * sample_count + index] =
        gpu_random_in_hemisphere(normal, seed);
}

int main() {
    GPUVec3* device_results = nullptr;

    std::vector<GPUVec3> host_results(
        3 * SAMPLE_COUNT
    );

    cudaError_t error = cudaMalloc(
        &device_results,
        3 * SAMPLE_COUNT * sizeof(GPUVec3)
    );

    if (error != cudaSuccess) {
        std::cerr << "cudaMalloc failed: "
                  << cudaGetErrorString(error) << '\n';
        return 1;
    }

    constexpr int THREADS_PER_BLOCK = 256;

    int blocks =
        (SAMPLE_COUNT + THREADS_PER_BLOCK - 1) /
        THREADS_PER_BLOCK;

    test_gpu_sampling<<<blocks, THREADS_PER_BLOCK>>>(
        device_results,
        SAMPLE_COUNT
    );

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        std::cerr << "Kernel failed: "
                  << cudaGetErrorString(error) << '\n';
        cudaFree(device_results);
        return 1;
    }

    error = cudaMemcpy(
        host_results.data(),
        device_results,
        3 * SAMPLE_COUNT * sizeof(GPUVec3),
        cudaMemcpyDeviceToHost
    );

    cudaFree(device_results);

    if (error != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: "
                  << cudaGetErrorString(error) << '\n';
        return 1;
    }

    bool sphere_pass = true;
    bool unit_vector_pass = true;
    bool hemisphere_pass = true;

    for (int i = 0; i < SAMPLE_COUNT; i++) {
        float sphere_length_squared =
            gpu_length_squared(host_results[i]);

        float unit_length =
            gpu_length(host_results[SAMPLE_COUNT + i]);

        float hemisphere_dot =
            gpu_dot(
                host_results[2 * SAMPLE_COUNT + i],
                GPUVec3{0.0f, 1.0f, 0.0f}
            );

        if (sphere_length_squared >= 1.0f) {
            sphere_pass = false;
        }

        if (fabsf(unit_length - 1.0f) > 1e-4f) {
            unit_vector_pass = false;
        }

        if (hemisphere_dot < 0.0f) {
            hemisphere_pass = false;
        }
    }

    std::cout << "Random points inside unit sphere: "
              << (sphere_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "Random unit vectors: "
              << (unit_vector_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "Random hemisphere vectors: "
              << (hemisphere_pass ? "PASS" : "FAIL") << '\n';

    bool all_pass =
        sphere_pass &&
        unit_vector_pass &&
        hemisphere_pass;

    std::cout << "GPU random sampling: "
              << (all_pass ? "PASS" : "FAIL") << '\n';

    return all_pass ? 0 : 1;
}