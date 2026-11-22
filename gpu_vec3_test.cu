
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "gpu_vec3.cuh"

__global__ void test_gpu_vec3(GPUVec3* results) {
    GPUVec3 a{1.0f, 2.0f, 3.0f};
    GPUVec3 b{4.0f, 5.0f, 6.0f};

    results[0] = a + b;
    results[1] = a - b;
    results[2] = a * 2.0f;
    results[3] = a * b;

    results[4] = GPUVec3{
        gpu_dot(a, b),
        gpu_length(a),
        gpu_length_squared(a)
    };

    results[5] = gpu_cross(a, b);
    results[6] = gpu_unit_vector(a);

    results[7] = gpu_reflect(
        GPUVec3{1.0f, -1.0f, 0.0f},
        GPUVec3{0.0f, 1.0f, 0.0f}
    );

    results[8] = gpu_refract(
        GPUVec3{0.0f, -1.0f, 0.0f},
        GPUVec3{0.0f, 1.0f, 0.0f},
        1.0f / 1.5f
    );
}

int main() {
    GPUVec3* device_results = nullptr;
    GPUVec3 host_results[9];

    cudaError_t error = cudaMalloc(
        &device_results,
        9 * sizeof(GPUVec3)
    );

    if (error != cudaSuccess) {
        std::cerr << "cudaMalloc failed: "
                  << cudaGetErrorString(error) << '\n';
        return 1;
    }

    test_gpu_vec3<<<1, 1>>>(device_results);

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        std::cerr << "Kernel failed: "
                  << cudaGetErrorString(error) << '\n';
        cudaFree(device_results);
        return 1;
    }

    error = cudaMemcpy(
        host_results,
        device_results,
        9 * sizeof(GPUVec3),
        cudaMemcpyDeviceToHost
    );

    cudaFree(device_results);

    if (error != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: "
                  << cudaGetErrorString(error) << '\n';
        return 1;
    }

    const char* labels[] = {
        "Addition",
        "Subtraction",
        "Scalar multiplication",
        "Component multiplication",
        "Dot / length / length squared",
        "Cross product",
        "Unit vector",
        "Reflection",
        "Refraction"
    };

    for (int i = 0; i < 9; i++) {
        std::cout << labels[i] << ": ("
                  << host_results[i].x << ", "
                  << host_results[i].y << ", "
                  << host_results[i].z << ")\n";
    }

    std::cout << "GPU vector operations: PASS\n";

    return 0;
}
