
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "gpu_random.cuh"

constexpr int SAMPLE_COUNT = 1 << 20;
constexpr int DRAWS_PER_SAMPLE = 4;

__global__ void generate_random_samples(float* values) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= SAMPLE_COUNT) {
        return;
    }

    // Hash consecutive sample indices before initializing Xorshift32.
    unsigned int seed = gpu_hash_seed(
        static_cast<unsigned int>(index + 1)
    );

    for (int draw = 0; draw < DRAWS_PER_SAMPLE; draw++) {
        values[index * DRAWS_PER_SAMPLE + draw] =
            gpu_random_float(seed);
    }
}

int main() {
    float* device_values = nullptr;

    std::vector<float> values(
        SAMPLE_COUNT * DRAWS_PER_SAMPLE
    );

    cudaError_t error = cudaMalloc(
        reinterpret_cast<void**>(&device_values),
        values.size() * sizeof(float)
    );

    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    constexpr int threads_per_block = 256;

    constexpr int blocks =
        (SAMPLE_COUNT + threads_per_block - 1) /
        threads_per_block;

    generate_random_samples<<<blocks, threads_per_block>>>(
        device_values
    );

    error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    error = cudaMemcpy(
        values.data(),
        device_values,
        values.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    cudaFree(device_values);

    constexpr int BIN_COUNT = 20;

    for (int draw = 0; draw < DRAWS_PER_SAMPLE; draw++) {
        double sum = 0.0;
        double sum_squared = 0.0;

        float minimum = 1.0f;
        float maximum = 0.0f;

        int histogram[BIN_COUNT] = {};

        double adjacent_product_sum = 0.0;
        double previous_value_sum = 0.0;
        double current_value_sum = 0.0;

        for (int i = 0; i < SAMPLE_COUNT; i++) {
            float value =
                values[i * DRAWS_PER_SAMPLE + draw];

            sum += value;
            sum_squared += value * value;

            if (value < minimum) minimum = value;
            if (value > maximum) maximum = value;

            int bin = static_cast<int>(value * BIN_COUNT);

            if (bin >= BIN_COUNT) {
                bin = BIN_COUNT - 1;
            }

            histogram[bin]++;

            if (i > 0) {
                float previous =
                    values[(i - 1) * DRAWS_PER_SAMPLE + draw];

                adjacent_product_sum += previous * value;
                previous_value_sum += previous;
                current_value_sum += value;
            }
        }

        double mean = sum / SAMPLE_COUNT;

        double variance =
            sum_squared / SAMPLE_COUNT - mean * mean;

        double covariance =
            adjacent_product_sum / (SAMPLE_COUNT - 1)
            - (previous_value_sum / (SAMPLE_COUNT - 1))
            * (current_value_sum / (SAMPLE_COUNT - 1));

        double correlation = covariance / variance;

        std::cout << "\nDraw " << draw + 1 << "\n";
        std::cout << "Mean:        " << mean << "\n";
        std::cout << "Variance:    " << variance << "\n";
        std::cout << "Minimum:     " << minimum << "\n";
        std::cout << "Maximum:     " << maximum << "\n";
        std::cout << "Adjacent correlation: "
                  << correlation << "\n";

        std::cout << "Histogram:\n";

        for (int bin = 0; bin < BIN_COUNT; bin++) {
            std::cout
                << bin * 5 << "% - " << (bin + 1) * 5
                << "%: " << histogram[bin] << "\n";
        }
    }

    return 0;
}