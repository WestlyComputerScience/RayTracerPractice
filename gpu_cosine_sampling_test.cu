
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "gpu_sampling.cuh"

constexpr int SAMPLE_COUNT = 16384;

__global__ void test_gpu_cosine_sampling(
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
        gpu_random_cosine_direction(normal, seed);
}

int main() {
    GPUVec3* device_results = nullptr;

    std::vector<GPUVec3> host_results(SAMPLE_COUNT);

    cudaError_t error = cudaMalloc(
        &device_results,
        SAMPLE_COUNT * sizeof(GPUVec3)
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

    test_gpu_cosine_sampling<<<blocks, THREADS_PER_BLOCK>>>(
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
        SAMPLE_COUNT * sizeof(GPUVec3),
        cudaMemcpyDeviceToHost
    );

    cudaFree(device_results);

    if (error != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: "
                  << cudaGetErrorString(error) << '\n';
        return 1;
    }

    bool unit_length_pass = true;
    bool hemisphere_pass = true;

    double cosine_sum = 0.0;

    for (const GPUVec3& direction : host_results) {
        float length = gpu_length(direction);

        if (fabsf(length - 1.0f) > 1e-4f) {
            unit_length_pass = false;
        }

        float cosine = direction.y;

        if (cosine < 0.0f) {
            hemisphere_pass = false;
        }

        cosine_sum += cosine;
    }

    double average_cosine =
        cosine_sum / SAMPLE_COUNT;

    bool distribution_pass =
        fabs(average_cosine - (2.0 / 3.0)) < 0.015;

    std::cout << "Unit-length directions: "
              << (unit_length_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "Upper hemisphere directions: "
              << (hemisphere_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "Average cosine: "
              << average_cosine << '\n';

    std::cout << "Cosine-weighted distribution: "
              << (distribution_pass ? "PASS" : "FAIL") << '\n';

    bool all_pass =
        unit_length_pass &&
        hemisphere_pass &&
        distribution_pass;

    std::cout << "GPU cosine sampling: "
              << (all_pass ? "PASS" : "FAIL") << '\n';

    return all_pass ? 0 : 1;
}
