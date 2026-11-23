
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "gpu_material.cuh"

__global__ void test_gpu_metal(
    float* results
) {
    GPUMaterial material{};

    material.type = GPUMaterialType::Metal;

    material.albedo_r = 0.8f;
    material.albedo_g = 0.4f;
    material.albedo_b = 0.2f;

    // Zero fuzz gives a deterministic perfect reflection.
    material.roughness = 0.0f;

    // Deliberately use a non-unit incoming direction.
    GPURay ray{
        0.0f, 1.0f, 0.0f,
        0.0f, -2.0f, 0.0f
    };

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

    results[5] = scatter.attenuation.x;
    results[6] = scatter.attenuation.y;
    results[7] = scatter.attenuation.z;
}

__global__ void test_gpu_dielectric(
    float* results
) {
    GPUMaterial material{};

    material.type = GPUMaterialType::Dielectric;
    material.refractive_index = 1.5f;

    // Ray is traveling from inside the dielectric.
    // Its angle causes total internal reflection.
    GPURay ray{
        0.0f, 0.0f, 0.0f,
        0.9f, 0.4358899f, 0.0f
    };

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

    results[5] = scatter.attenuation.x;
    results[6] = scatter.attenuation.y;
    results[7] = scatter.attenuation.z;
}

bool approximately_equal(
    float a,
    float b,
    float epsilon = 1e-5f
) {
    return std::fabs(a - b) <= epsilon;
}

int main() {
    float* device_metal = nullptr;
    float* device_dielectric = nullptr;

    float metal[8] = {};
    float dielectric[8] = {};

    cudaError_t error = cudaMalloc(
        &device_metal,
        sizeof(metal)
    );

    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << '\n';
        return 1;
    }

    error = cudaMalloc(
        &device_dielectric,
        sizeof(dielectric)
    );

    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << '\n';
        cudaFree(device_metal);
        return 1;
    }

    test_gpu_metal<<<1, 1>>>(device_metal);
    test_gpu_dielectric<<<1, 1>>>(device_dielectric);

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        std::cerr << "Kernel failure: "
                  << cudaGetErrorString(error) << '\n';

        cudaFree(device_metal);
        cudaFree(device_dielectric);

        return 1;
    }

    cudaMemcpy(
        metal,
        device_metal,
        sizeof(metal),
        cudaMemcpyDeviceToHost
    );

    cudaMemcpy(
        dielectric,
        device_dielectric,
        sizeof(dielectric),
        cudaMemcpyDeviceToHost
    );

    cudaFree(device_metal);
    cudaFree(device_dielectric);

    bool metal_pass =
        approximately_equal(metal[0], 1.0f) &&
        approximately_equal(metal[1], 1.0f) &&
        approximately_equal(metal[2], 0.0f) &&
        approximately_equal(metal[3], 1.0f) &&
        approximately_equal(metal[4], 0.0f) &&
        approximately_equal(metal[5], 0.8f) &&
        approximately_equal(metal[6], 0.4f) &&
        approximately_equal(metal[7], 0.2f);

    bool dielectric_pass =
        approximately_equal(dielectric[0], 1.0f) &&
        approximately_equal(dielectric[1], 1.0f) &&
        approximately_equal(dielectric[2], 0.9f) &&
        approximately_equal(dielectric[3], -0.4358899f) &&
        approximately_equal(dielectric[4], 0.0f) &&
        approximately_equal(dielectric[5], 1.0f) &&
        approximately_equal(dielectric[6], 1.0f) &&
        approximately_equal(dielectric[7], 1.0f);

    std::cout << "Metal perfect reflection: "
              << (metal_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "Dielectric total internal reflection: "
              << (dielectric_pass ? "PASS" : "FAIL") << '\n';

    if (!metal_pass) {
        std::cout << "\nMetal results:\n";
        for (float value : metal) {
            std::cout << value << '\n';
        }
    }

    if (!dielectric_pass) {
        std::cout << "\nDielectric results:\n";
        for (float value : dielectric) {
            std::cout << value << '\n';
        }
    }

    bool all_passed = metal_pass && dielectric_pass;

    std::cout << "\nGPU material scatter tests: "
              << (all_passed ? "PASS" : "FAIL") << '\n';

    return all_passed ? 0 : 1;
}