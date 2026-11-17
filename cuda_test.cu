
#include <iostream>
#include <cuda_runtime.h>

__global__ void test_kernel(int* result) {
    int index = threadIdx.x;
    result[index] = index * index;
}

int main() {
    int* device_result = nullptr;
    int host_result[10];

    cudaError_t error = cudaMalloc(
        &device_result,
        10 * sizeof(int)
    );

    if (error != cudaSuccess) {
        std::cerr << "cudaMalloc failed: "
                  << cudaGetErrorString(error) << '\n';
        return 1;
    }

    test_kernel<<<1, 10>>>(device_result);

    error = cudaGetLastError();

    if (error != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                  << cudaGetErrorString(error) << '\n';
        cudaFree(device_result);
        return 1;
    }

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        std::cerr << "Kernel execution failed: "
                  << cudaGetErrorString(error) << '\n';
        cudaFree(device_result);
        return 1;
    }

    error = cudaMemcpy(
        host_result,
        device_result,
        10 * sizeof(int),
        cudaMemcpyDeviceToHost
    );

    if (error != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: "
                  << cudaGetErrorString(error) << '\n';
        cudaFree(device_result);
        return 1;
    }

    std::cout << "CUDA kernel results:\n";

    for (int i = 0; i < 10; i++) {
        std::cout << host_result[i] << '\n';
    }

    cudaFree(device_result);

    return 0;
}
