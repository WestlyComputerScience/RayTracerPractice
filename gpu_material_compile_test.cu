
#include <cuda_runtime.h>

#include <iostream>

#include "gpu_material.cuh"

constexpr int MATERIAL_COUNT = 6;
constexpr int RESULT_COUNT = 3;

__global__ void test_gpu_materials(int* results) {
    GPURay ray{
        0.0f, 1.0f, 0.0f,
        0.0f, -1.0f, 0.0f
    };

    GPUTraversalResult hit{};

    hit.hit = 1;
    hit.position_y = 0.0f;
    hit.normal_y = 1.0f;

    unsigned int seed = 12345u;

    for (int i = 0; i < MATERIAL_COUNT; i++) {
        GPUMaterial material{};

        material.type = static_cast<GPUMaterialType>(i);

        material.albedo_r = 0.8f;
        material.albedo_g = 0.4f;
        material.albedo_b = 0.2f;

        material.roughness = 0.5f;
        material.metalness = 0.5f;
        material.refractive_index = 1.5f;

        material.emission_r = 5.0f;
        material.emission_g = 5.0f;
        material.emission_b = 5.0f;

        GPUScatterRecord scatter{};

        bool scattered = gpu_material_scatter(
            material,
            ray,
            hit,
            seed,
            scatter
        );

        int offset = i * RESULT_COUNT;

        results[offset] = scattered ? 1 : 0;
        results[offset + 1] = static_cast<int>(scatter.pdf_type);
        results[offset + 2] = scatter.skip_pdf ? 1 : 0;
    }
}

int main() {
    int* device_results = nullptr;

    int host_results[MATERIAL_COUNT * RESULT_COUNT] = {};

    cudaError_t error = cudaMalloc(
        &device_results,
        sizeof(host_results)
    );

    if (error != cudaSuccess) {
        std::cerr << "cudaMalloc failed: "
                  << cudaGetErrorString(error) << '\n';
        return 1;
    }

    test_gpu_materials<<<1, 1>>>(device_results);

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
        sizeof(host_results),
        cudaMemcpyDeviceToHost
    );

    cudaFree(device_results);

    if (error != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: "
                  << cudaGetErrorString(error) << '\n';
        return 1;
    }

    const int expected_scattered[MATERIAL_COUNT] = {
        1,  // Lambertian
        1,  // Metal
        1,  // Dielectric
        0,  // DiffuseLight
        1,  // Isotropic
        1   // GGX
    };

    const int expected_pdf[MATERIAL_COUNT] = {
        static_cast<int>(GPUScatterPdfType::Cosine),
        static_cast<int>(GPUScatterPdfType::None),
        static_cast<int>(GPUScatterPdfType::None),
        static_cast<int>(GPUScatterPdfType::None),
        static_cast<int>(GPUScatterPdfType::Sphere),
        static_cast<int>(GPUScatterPdfType::GGX)
    };

    const int expected_skip_pdf[MATERIAL_COUNT] = {
        0,  // Lambertian
        1,  // Metal
        1,  // Dielectric
        0,  // DiffuseLight
        0,  // Isotropic
        0   // GGX
    };

    const char* material_names[MATERIAL_COUNT] = {
        "Lambertian",
        "Metal",
        "Dielectric",
        "DiffuseLight",
        "Isotropic",
        "GGX"
    };

    bool all_passed = true;

    for (int i = 0; i < MATERIAL_COUNT; i++) {
        int offset = i * RESULT_COUNT;

        bool passed =
            host_results[offset] == expected_scattered[i] &&
            host_results[offset + 1] == expected_pdf[i] &&
            host_results[offset + 2] == expected_skip_pdf[i];

        std::cout
            << material_names[i]
            << ": "
            << (passed ? "PASS" : "FAIL")
            << '\n';

        if (!passed) {
            std::cout
                << "  Scattered: "
                << host_results[offset]
                << " (expected " << expected_scattered[i] << ")\n"
                << "  PDF type: "
                << host_results[offset + 1]
                << " (expected " << expected_pdf[i] << ")\n"
                << "  Skip PDF: "
                << host_results[offset + 2]
                << " (expected " << expected_skip_pdf[i] << ")\n";

            all_passed = false;
        }
    }

    std::cout
        << "\nGPU material behavior tests: "
        << (all_passed ? "PASS" : "FAIL")
        << '\n';

    return all_passed ? 0 : 1;
}
