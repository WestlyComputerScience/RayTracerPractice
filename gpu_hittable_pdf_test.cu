#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "gpu_pdf.cuh"

__global__ void test_gpu_hittable_pdf(
    int* results,
    float* generated_directions
) {
    GPUPrimitive primitives[2]{};

    for (int i = 0; i < 2; i++) {
        primitives[i].type = GPUPrimitiveType::Quad;

        // Q
        primitives[i].parameters[0] = i * 4.0f;
        primitives[i].parameters[1] = 0.0f;
        primitives[i].parameters[2] = 0.0f;

        // u
        primitives[i].parameters[3] = 2.0f;
        primitives[i].parameters[4] = 0.0f;
        primitives[i].parameters[5] = 0.0f;

        // v
        primitives[i].parameters[6] = 0.0f;
        primitives[i].parameters[7] = 0.0f;
        primitives[i].parameters[8] = 2.0f;

        // Normal
        primitives[i].parameters[9] = 0.0f;
        primitives[i].parameters[10] = -1.0f;
        primitives[i].parameters[11] = 0.0f;

        // D and area
        primitives[i].parameters[12] = 0.0f;
        primitives[i].parameters[13] = 4.0f;
    }

    int light_indices[2] = {0, 1};

    GPUHittablePdf pdf{
        1.0f,
        1.0f,
        -2.0f
    };

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int seed = 12345u + index * 7919u;

    // Generate a direction toward one of the two lights.
    float dx, dy, dz;

    bool generated = gpu_hittable_pdf_generate(
        primitives,
        light_indices,
        2,
        pdf,
        seed,
        dx,
        dy,
        dz
    );

    if (!generated) {
        atomicAdd(&results[0], 1);
        return;
    }

    // Determine which quad was sampled.
    float point_x = pdf.origin_x + dx;

    if (point_x >= 0.0f && point_x <= 2.0f) {
        atomicAdd(&results[1], 1);
    }
    else if (point_x >= 4.0f && point_x <= 6.0f) {
        atomicAdd(&results[2], 1);
    }
    else {
        atomicAdd(&results[3], 1);
    }

    // Save the first generated direction.
    if (index == 0) {
        generated_directions[0] = dx;
        generated_directions[1] = dy;
        generated_directions[2] = dz;
    }

    // Evaluate the PDF for a direction that intersects
    // the first quad at (1, 0, 0).
    float test_pdf = gpu_hittable_pdf_value(
        primitives,
        light_indices,
        2,
        pdf,
        0.0f,
        -1.0f,
        2.0f
    );

    if (test_pdf <= 0.0f) {
        atomicAdd(&results[4], 1);
    }

    // Empty light list should safely return zero.
    float empty_pdf = gpu_hittable_pdf_value(
        primitives,
        light_indices,
        0,
        pdf,
        0.0f,
        -1.0f,
        2.0f
    );

    if (empty_pdf != 0.0f) {
        atomicAdd(&results[5], 1);
    }

    float empty_dx, empty_dy, empty_dz;

    bool empty_generated = gpu_hittable_pdf_generate(
        primitives,
        light_indices,
        0,
        pdf,
        seed,
        empty_dx,
        empty_dy,
        empty_dz
    );

    if (empty_generated) {
        atomicAdd(&results[6], 1);
    }
}

int main() {
    constexpr int sample_count = 10000;

    int host_results[7] = {};
    float host_direction[3] = {};

    int* device_results = nullptr;
    float* device_direction = nullptr;

    cudaError_t error = cudaMalloc(
        &device_results,
        7 * sizeof(int)
    );

    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << '\n';
        return 1;
    }

    error = cudaMalloc(
        &device_direction,
        3 * sizeof(float)
    );

    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << '\n';
        cudaFree(device_results);
        return 1;
    }

    cudaMemset(device_results, 0, 7 * sizeof(int));

    test_gpu_hittable_pdf<<<sample_count, 1>>>(
        device_results,
        device_direction
    );

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        std::cerr << "Kernel failed: "
                  << cudaGetErrorString(error) << '\n';
        cudaFree(device_results);
        cudaFree(device_direction);
        return 1;
    }

    cudaMemcpy(
        host_results,
        device_results,
        7 * sizeof(int),
        cudaMemcpyDeviceToHost
    );

    cudaMemcpy(
        host_direction,
        device_direction,
        3 * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    cudaFree(device_results);
    cudaFree(device_direction);

    bool generation_pass = host_results[0] == 0;
    bool first_light_pass = host_results[1] > 0;
    bool second_light_pass = host_results[2] > 0;
    bool invalid_direction_pass = host_results[3] == 0;
    bool pdf_value_pass = host_results[4] == 0;
    bool empty_pdf_pass = host_results[5] == 0;
    bool empty_generation_pass = host_results[6] == 0;

    std::cout << "Light generation: "
              << (generation_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "First light selected: "
              << (first_light_pass ? "PASS" : "FAIL")
              << " (" << host_results[1] << ")\n";

    std::cout << "Second light selected: "
              << (second_light_pass ? "PASS" : "FAIL")
              << " (" << host_results[2] << ")\n";

    std::cout << "Generated directions valid: "
              << (invalid_direction_pass ? "PASS" : "FAIL")
              << " (" << host_results[3] << " invalid)\n";

    std::cout << "Hittable PDF value: "
              << (pdf_value_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "Empty light PDF: "
              << (empty_pdf_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "Empty light generation: "
              << (empty_generation_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "First direction: ("
              << host_direction[0] << ", "
              << host_direction[1] << ", "
              << host_direction[2] << ")\n";

    bool all_pass =
        generation_pass &&
        first_light_pass &&
        second_light_pass &&
        invalid_direction_pass &&
        pdf_value_pass &&
        empty_pdf_pass &&
        empty_generation_pass;

    std::cout << "GPU HittablePdf tests: "
              << (all_pass ? "PASS" : "FAIL") << '\n';

    return all_pass ? 0 : 1;
}