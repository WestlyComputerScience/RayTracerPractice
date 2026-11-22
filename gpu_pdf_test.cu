
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "gpu_pdf.cuh"

__global__ void test_gpu_pdfs(float* results) {
    unsigned int seed = 12345u;

    GPUCosinePdf cosine_pdf{
        0.0f, 1.0f, 0.0f
    };

    // Sphere PDF evaluation.
    results[0] = gpu_sphere_pdf_value();

    // Cosine PDF evaluation.
    results[1] = gpu_cosine_pdf_value(
        cosine_pdf,
        0.0f, 1.0f, 0.0f
    );

    // Opposite hemisphere should have zero density.
    results[2] = gpu_cosine_pdf_value(
        cosine_pdf,
        0.0f, -1.0f, 0.0f
    );

    // Generate a sphere PDF sample.
    gpu_sphere_pdf_generate(
        seed,
        results[3],
        results[4],
        results[5]
    );

    // Generate a cosine PDF sample.
    gpu_cosine_pdf_generate(
        cosine_pdf,
        seed,
        results[6],
        results[7],
        results[8]
    );
}

int main() {
    float host_results[9] = {};
    float* device_results = nullptr;

    cudaError_t error = cudaMalloc(
        &device_results,
        9 * sizeof(float)
    );

    if (error != cudaSuccess) {
        std::cerr << "cudaMalloc failed: "
                  << cudaGetErrorString(error) << '\n';
        return 1;
    }

    test_gpu_pdfs<<<1, 1>>>(device_results);

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
        9 * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    cudaFree(device_results);

    if (error != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: "
                  << cudaGetErrorString(error) << '\n';
        return 1;
    }

    constexpr float PI =
        3.14159265358979323846f;

    bool sphere_value_pass =
        fabsf(host_results[0] - 1.0f / (4.0f * PI)) < 1e-6f;

    bool cosine_value_pass =
        fabsf(host_results[1] - 1.0f / PI) < 1e-6f;

    bool opposite_value_pass =
        host_results[2] == 0.0f;

    float sphere_length = sqrtf(
        host_results[3] * host_results[3] +
        host_results[4] * host_results[4] +
        host_results[5] * host_results[5]
    );

    bool sphere_generation_pass =
        fabsf(sphere_length - 1.0f) < 1e-4f;

    float cosine_length = sqrtf(
        host_results[6] * host_results[6] +
        host_results[7] * host_results[7] +
        host_results[8] * host_results[8]
    );

    bool cosine_generation_pass =
        fabsf(cosine_length - 1.0f) < 1e-4f &&
        host_results[7] >= 0.0f;

    std::cout << "Sphere PDF value: "
              << (sphere_value_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "Cosine PDF value: "
              << (cosine_value_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "Opposite hemisphere PDF: "
              << (opposite_value_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "Sphere PDF generation: "
              << (sphere_generation_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "Cosine PDF generation: "
              << (cosine_generation_pass ? "PASS" : "FAIL") << '\n';

    bool all_pass =
        sphere_value_pass &&
        cosine_value_pass &&
        opposite_value_pass &&
        sphere_generation_pass &&
        cosine_generation_pass;

    std::cout << "GPU PDF tests: "
              << (all_pass ? "PASS" : "FAIL") << '\n';

    return all_pass ? 0 : 1;
}
