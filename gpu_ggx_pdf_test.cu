
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "gpu_pdf.cuh"


__global__ void test_gpu_ggx_pdf(
    int* results,
    float* values
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int seed = 12345u + index * 7919u;

    // Match the CPU test's normal and grazing-angle view.
    GPUVec3 normal = gpu_unit_vector(
        GPUVec3{0.0f, 1.0f, 0.0f}
    );

    GPUVec3 view = gpu_unit_vector(
        GPUVec3{0.9f, 0.2f, -0.2f}
    );

    GPUGGXPdf pdf{};

    pdf.normal_x = normal.x;
    pdf.normal_y = normal.y;
    pdf.normal_z = normal.z;

    pdf.view_x = view.x;
    pdf.view_y = view.y;
    pdf.view_z = view.z;

    pdf.roughness = 0.2f;

    // Match the CPU GGXPdf constructor.
    GPUVec3 reference =
        fabsf(normal.x) > 0.9f
            ? GPUVec3{0.0f, 1.0f, 0.0f}
            : GPUVec3{1.0f, 0.0f, 0.0f};

    GPUVec3 tangent = gpu_unit_vector(
        gpu_cross(reference, normal)
    );

    GPUVec3 bitangent = gpu_cross(normal, tangent);

    pdf.tangent_x = tangent.x;
    pdf.tangent_y = tangent.y;
    pdf.tangent_z = tangent.z;

    pdf.bitangent_x = bitangent.x;
    pdf.bitangent_y = bitangent.y;
    pdf.bitangent_z = bitangent.z;

    // Test the GGX distribution.
    float D = gpu_ggx_distribution(
        pdf,
        normal.x,
        normal.y,
        normal.z
    );

    if (!isfinite(D) || D <= 0.0f) {
        atomicAdd(&results[0], 1);
    }

    // Test the Smith geometry function.
    float n_dot_v = gpu_dot(normal, view);

    float G1 = gpu_ggx_geometry_smith_g1(
        pdf,
        n_dot_v
    );

    if (!isfinite(G1) || G1 <= 0.0f || G1 > 1.00001f) {
        atomicAdd(&results[1], 1);
    }

    // Test VNDF sampling and reflected directions.
    float dx, dy, dz;

    gpu_ggx_pdf_generate(
        pdf,
        seed,
        dx,
        dy,
        dz
    );

    float direction_length = sqrtf(
        dx * dx + dy * dy + dz * dz
    );

    if (
        !isfinite(direction_length) ||
        fabsf(direction_length - 1.0f) > 1e-4f
    ) {
        atomicAdd(&results[2], 1);
    }

    float n_dot_l =
        dx * normal.x +
        dy * normal.y +
        dz * normal.z;

    // Below-surface directions are possible with VNDF sampling.
    if (n_dot_l <= 0.0f) {
        atomicAdd(&results[3], 1);
    }

    // PDF evaluation must agree with the sampled direction.
    float pdf_value = gpu_ggx_pdf_value(
        pdf,
        dx,
        dy,
        dz
    );

    if (n_dot_l <= 0.0f) {
        if (pdf_value != 0.0f) {
            atomicAdd(&results[4], 1);
        }
    }
    else {
        if (!isfinite(pdf_value) || pdf_value <= 0.0f) {
            atomicAdd(&results[4], 1);
        }
    }

    // Test rejection of a direction below the surface.
    float below_surface = gpu_ggx_pdf_value(
        pdf,
        0.0f,
        -1.0f,
        0.0f
    );

    if (below_surface != 0.0f) {
        atomicAdd(&results[5], 1);
    }

    // Test rejection of a zero direction.
    float zero_direction = gpu_ggx_pdf_value(
        pdf,
        0.0f,
        0.0f,
        0.0f
    );

    if (zero_direction != 0.0f) {
        atomicAdd(&results[6], 1);
    }

    if (index == 0) {
        values[0] = D;
        values[1] = G1;
        values[2] = pdf_value;
        values[3] = n_dot_v;
    }
}

int main() {
    constexpr int sample_count = 10000;

    int host_results[7] = {};
    float host_values[4] = {};

    int* device_results = nullptr;
    float* device_values = nullptr;

    cudaError_t error = cudaMalloc(
        &device_results,
        7 * sizeof(int)
    );

    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << '\n';
        return 1;
    }

    error = cudaMalloc(
        &device_values,
        4 * sizeof(float)
    );

    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << '\n';
        cudaFree(device_results);
        return 1;
    }

    cudaMemset(device_results, 0, 7 * sizeof(int));

    test_gpu_ggx_pdf<<<sample_count, 1>>>(
        device_results,
        device_values
    );

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        std::cerr << "Kernel failed: "
                  << cudaGetErrorString(error) << '\n';

        cudaFree(device_results);
        cudaFree(device_values);
        return 1;
    }

    error = cudaMemcpy(
        host_results,
        device_results,
        7 * sizeof(int),
        cudaMemcpyDeviceToHost
    );

    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << '\n';
        cudaFree(device_results);
        cudaFree(device_values);
        return 1;
    }

    error = cudaMemcpy(
        host_values,
        device_values,
        4 * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    cudaFree(device_results);
    cudaFree(device_values);

    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << '\n';
        return 1;
    }

    bool distribution_pass = host_results[0] == 0;
    bool geometry_pass = host_results[1] == 0;
    bool direction_length_pass = host_results[2] == 0;
    bool pdf_consistency_pass = host_results[4] == 0;
    bool below_surface_pass = host_results[5] == 0;
    bool zero_direction_pass = host_results[6] == 0;

    std::cout << "GGX distribution: "
              << (distribution_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "Smith geometry: "
              << (geometry_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "Generated direction length: "
              << (direction_length_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "Below-surface VNDF samples: "
          << host_results[3]
          << " (possible with this sampling method)\n";

    std::cout << "GGX PDF consistency: "
            << (pdf_consistency_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "\nGGX values:\n";
    std::cout << "D: " << host_values[0] << '\n';
    std::cout << "G1: " << host_values[1] << '\n';
    std::cout << "PDF: " << host_values[2] << '\n';
    std::cout << "N dot V: " << host_values[3] << '\n';

    bool all_pass =
        distribution_pass &&
        geometry_pass &&
        direction_length_pass &&
        pdf_consistency_pass &&
        below_surface_pass &&
        zero_direction_pass;

    std::cout << "\nGPU GGX PDF tests: "
              << (all_pass ? "PASS" : "FAIL") << '\n';

    return all_pass ? 0 : 1;
}