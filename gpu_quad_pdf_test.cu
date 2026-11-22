
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "gpu_pdf.cuh"

__global__ void test_gpu_quad_pdf(float* result) {
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

    // Normal = (0, 0, 1)
    quad.parameters[9] = 0.0f;
    quad.parameters[10] = 0.0f;
    quad.parameters[11] = 1.0f;

    // D = 0, area = 4
    quad.parameters[12] = 0.0f;
    quad.parameters[13] = 4.0f;

    GPUHittablePdf pdf{
        1.0f,
        1.0f,
        -2.0f
    };

    // Direction toward the center of the quad.
    result[0] = gpu_quad_pdf_value(
        quad,
        pdf,
        0.0f,
        0.0f,
        2.0f
    );

    // Direction pointing away from the quad.
    result[1] = gpu_quad_pdf_value(
        quad,
        pdf,
        0.0f,
        0.0f,
        -2.0f
    );

    // Direction that misses the parallelogram.
    result[2] = gpu_quad_pdf_value(
        quad,
        pdf,
        4.0f,
        0.0f,
        2.0f
    );

    // Direction parallel to the quad.
    result[3] = gpu_quad_pdf_value(
        quad,
        pdf,
        1.0f,
        0.0f,
        0.0f
    );
}

int main() {
    float host_results[4] = {};
    float* device_results = nullptr;

    cudaError_t error = cudaMalloc(
        &device_results,
        4 * sizeof(float)
    );

    if (error != cudaSuccess) {
        std::cerr << "cudaMalloc failed: "
                  << cudaGetErrorString(error) << '\n';
        return 1;
    }

    test_gpu_quad_pdf<<<1, 1>>>(device_results);

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
        4 * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    cudaFree(device_results);

    if (error != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: "
                  << cudaGetErrorString(error) << '\n';
        return 1;
    }

    bool valid_pdf =
        fabsf(host_results[0] - 1.0f) < 1e-5f;

    bool away_pdf =
        host_results[1] == 0.0f;

    bool miss_pdf =
        host_results[2] == 0.0f;

    bool parallel_pdf =
        host_results[3] == 0.0f;

    std::cout << "Quad PDF valid intersection: "
              << (valid_pdf ? "PASS" : "FAIL")
              << " (" << host_results[0] << ")\n";

    std::cout << "Quad PDF ray pointing away: "
              << (away_pdf ? "PASS" : "FAIL") << '\n';

    std::cout << "Quad PDF outside parallelogram: "
              << (miss_pdf ? "PASS" : "FAIL") << '\n';

    std::cout << "Quad PDF parallel ray: "
              << (parallel_pdf ? "PASS" : "FAIL") << '\n';

    bool all_pass =
        valid_pdf &&
        away_pdf &&
        miss_pdf &&
        parallel_pdf;

    std::cout << "GPU quad PDF tests: "
              << (all_pass ? "PASS" : "FAIL") << '\n';

    return all_pass ? 0 : 1;
}
