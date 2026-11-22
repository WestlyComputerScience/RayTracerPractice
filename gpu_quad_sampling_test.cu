
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "gpu_pdf.cuh"


__global__ void test_gpu_quad_sampling(
    int* results,
    float* first_direction
) {
    GPUPrimitive quad{};

    quad.type = GPUPrimitiveType::Quad;

    // Q = (0, 0, 0)
    quad.parameters[0] = 0.0f;
    quad.parameters[1] = 0.0f;
    quad.parameters[2] = 0.0f;

    // u = (2, 0, 0)
    quad.parameters[3] = 2.0f;
    quad.parameters[4] = 0.0f;
    quad.parameters[5] = 0.0f;

    // v = (0, 0, 2)
    quad.parameters[6] = 0.0f;
    quad.parameters[7] = 0.0f;
    quad.parameters[8] = 2.0f;

    // Normal = (0, -1, 0)
    quad.parameters[9] = 0.0f;
    quad.parameters[10] = -1.0f;
    quad.parameters[11] = 0.0f;

    // D = 0, area = 4
    quad.parameters[12] = 0.0f;
    quad.parameters[13] = 4.0f;

    GPUHittablePdf pdf{
        1.0f,
        1.0f,
        -2.0f
    };

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int seed = 12345u + index * 7919u;

    float dx, dy, dz;

    gpu_quad_pdf_generate(
        quad,
        pdf,
        seed,
        dx,
        dy,
        dz
    );

    float point_x = pdf.origin_x + dx;
    float point_y = pdf.origin_y + dy;
    float point_z = pdf.origin_z + dz;

    // All points must lie on y = 0.
    if (fabsf(point_y) > 1e-5f) {
        atomicAdd(&results[0], 1);
    }

    // All points must lie inside x,z in [0,2].
    if (
        point_x < -1e-5f || point_x > 2.00001f ||
        point_z < -1e-5f || point_z > 2.00001f
    ) {
        atomicAdd(&results[1], 1);
    }

    // Direction must be nonzero.
    float length_squared =
        dx * dx + dy * dy + dz * dz;

    if (length_squared <= 1e-8f) {
        atomicAdd(&results[2], 1);
    }

    if (index == 0) {
        first_direction[0] = dx;
        first_direction[1] = dy;
        first_direction[2] = dz;
    }
}

int main() {
    constexpr int sample_count = 1000;

    int host_results[3] = {};
    float host_first_direction[3] = {};

    int* device_results = nullptr;
    float* device_first_direction = nullptr;

    cudaError_t error = cudaMalloc(
        &device_results,
        3 * sizeof(int)
    );

    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << '\n';
        return 1;
    }

    error = cudaMalloc(
        &device_first_direction,
        3 * sizeof(float)
    );

    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << '\n';
        cudaFree(device_results);
        return 1;
    }

    cudaMemset(device_results, 0, 3 * sizeof(int));

    test_gpu_quad_sampling<<<sample_count, 1>>>(
        device_results,
        device_first_direction
    );

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        std::cerr << "Kernel failed: "
                  << cudaGetErrorString(error) << '\n';
        cudaFree(device_results);
        cudaFree(device_first_direction);
        return 1;
    }

    error = cudaMemcpy(
        host_results,
        device_results,
        3 * sizeof(int),
        cudaMemcpyDeviceToHost
    );

    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << '\n';
        cudaFree(device_results);
        cudaFree(device_first_direction);
        return 1;
    }

    error = cudaMemcpy(
        host_first_direction,
        device_first_direction,
        3 * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    cudaFree(device_results);
    cudaFree(device_first_direction);

    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << '\n';
        return 1;
    }

    bool plane_pass = host_results[0] == 0;
    bool bounds_pass = host_results[1] == 0;
    bool direction_pass = host_results[2] == 0;

    std::cout << "Quad plane sampling: "
              << (plane_pass ? "PASS" : "FAIL")
              << " (invalid: " << host_results[0] << ")\n";

    std::cout << "Quad boundary sampling: "
              << (bounds_pass ? "PASS" : "FAIL")
              << " (invalid: " << host_results[1] << ")\n";

    std::cout << "Nonzero directions: "
              << (direction_pass ? "PASS" : "FAIL")
              << " (invalid: " << host_results[2] << ")\n";

    std::cout << "First generated direction: ("
              << host_first_direction[0] << ", "
              << host_first_direction[1] << ", "
              << host_first_direction[2] << ")\n";

    bool all_pass =
        plane_pass &&
        bounds_pass &&
        direction_pass;

    std::cout << "GPU quad sampling tests: "
              << (all_pass ? "PASS" : "FAIL") << '\n';

    return all_pass ? 0 : 1;
}
