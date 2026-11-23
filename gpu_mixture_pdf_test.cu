
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "gpu_material.cuh"


__global__ void test_gpu_mixture_pdf(
    int* results,
    float* mixture_values
) {
    GPUPrimitive primitives[2]{};

    // Construct two quad lights.
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

    GPUHittablePdf hittable_pdf{
        1.0f,
        1.0f,
        -2.0f
    };

    GPUScatterRecord material_scatter{};

    material_scatter.pdf_type = GPUScatterPdfType::Cosine;

    material_scatter.cosine_pdf = GPUCosinePdf{
        0.0f,
        -1.0f,
        0.0f
    };

    GPUMixturePdf mixture{};

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int seed = 12345u + index * 7919u;

    float dx, dy, dz;

    // Test mixture generation.
    bool generated = gpu_mixture_pdf_generate(
        mixture,
        primitives,
        light_indices,
        2,
        hittable_pdf,
        material_scatter,
        seed,
        dx,
        dy,
        dz
    );

    if (!generated) {
        atomicAdd(&results[0], 1);
    }
    else {
        float length_squared =
            dx * dx + dy * dy + dz * dz;

        if (length_squared <= 1e-8f) {
            atomicAdd(&results[3], 1);
        }

        if (mixture.sample_hittable) {
            atomicAdd(&results[1], 1);

            float point_x = hittable_pdf.origin_x + dx;

            if (
                !(
                    (point_x >= 0.0f && point_x <= 2.0f) ||
                    (point_x >= 4.0f && point_x <= 6.0f)
                )
            ) {
                atomicAdd(&results[4], 1);
            }
        }
        else {
            atomicAdd(&results[2], 1);

            float direction_length = sqrtf(length_squared);

            if (fabsf(direction_length - 1.0f) > 1e-4f) {
                atomicAdd(&results[3], 1);
            }

            float cosine =
                dx * material_scatter.cosine_pdf.normal_x +
                dy * material_scatter.cosine_pdf.normal_y +
                dz * material_scatter.cosine_pdf.normal_z;

            if (cosine < -1e-5f) {
                atomicAdd(&results[3], 1);
            }
        }
    }

    // Test mixture PDF evaluation.
    float test_dx = 0.0f;
    float test_dy = -1.0f;
    float test_dz = 2.0f;

    float hittable_value = gpu_hittable_pdf_value(
        primitives,
        light_indices,
        2,
        hittable_pdf,
        test_dx,
        test_dy,
        test_dz
    );

    float material_value = gpu_material_pdf_value(
        material_scatter,
        test_dx,
        test_dy,
        test_dz
    );

    float expected_value =
        0.5f * hittable_value +
        0.5f * material_value;

    float actual_value = gpu_mixture_pdf_value(
        mixture,
        primitives,
        light_indices,
        2,
        hittable_pdf,
        material_scatter,
        test_dx,
        test_dy,
        test_dz
    );

    if (fabsf(actual_value - expected_value) > 1e-5f) {
        atomicAdd(&results[5], 1);
    }

    if (index == 0) {
        mixture_values[0] = hittable_value;
        mixture_values[1] = material_value;
        mixture_values[2] = actual_value;
    }

    // Test mixture generation with no available lights.
    GPUMixturePdf empty_mixture{};

    unsigned int empty_seed = 54321u + index * 7919u;

    float empty_dx, empty_dy, empty_dz;

    bool empty_generated = gpu_mixture_pdf_generate(
        empty_mixture,
        nullptr,
        nullptr,
        0,
        hittable_pdf,
        material_scatter,
        empty_seed,
        empty_dx,
        empty_dy,
        empty_dz
    );

    if (empty_mixture.sample_hittable) {
        if (empty_generated) {
            atomicAdd(&results[6], 1);
        }

        if (
            empty_dx != 0.0f ||
            empty_dy != 0.0f ||
            empty_dz != 0.0f
        ) {
            atomicAdd(&results[6], 1);
        }
    }
    else {
        if (!empty_generated) {
            atomicAdd(&results[6], 1);
        }

        float empty_length = sqrtf(
            empty_dx * empty_dx +
            empty_dy * empty_dy +
            empty_dz * empty_dz
        );

        if (
            fabsf(empty_length - 1.0f) > 1e-4f ||
            empty_dy > 1e-5f
        ) {
            atomicAdd(&results[6], 1);
        }
    }
}

__global__ void test_gpu_mixture_material_types(
    int* results,
    float* pdf_values
) {
    GPUPrimitive primitives[2]{};

    for (int i = 0; i < 2; i++) {
        primitives[i].type = GPUPrimitiveType::Quad;

        primitives[i].parameters[0] = i * 4.0f;
        primitives[i].parameters[1] = 0.0f;
        primitives[i].parameters[2] = 0.0f;

        primitives[i].parameters[3] = 2.0f;
        primitives[i].parameters[4] = 0.0f;
        primitives[i].parameters[5] = 0.0f;

        primitives[i].parameters[6] = 0.0f;
        primitives[i].parameters[7] = 0.0f;
        primitives[i].parameters[8] = 2.0f;

        primitives[i].parameters[9] = 0.0f;
        primitives[i].parameters[10] = -1.0f;
        primitives[i].parameters[11] = 0.0f;

        primitives[i].parameters[12] = 0.0f;
        primitives[i].parameters[13] = 4.0f;
    }

    int light_indices[2] = {0, 1};

    GPUHittablePdf hittable_pdf{
        1.0f,
        1.0f,
        -2.0f
    };

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Test Sphere and GGX.
    for (int type_index = 0; type_index < 2; type_index++) {
        int base = type_index * 5;

        GPUScatterRecord scatter{};

        if (type_index == 0) {
            scatter.pdf_type = GPUScatterPdfType::Sphere;
        }
        else {
            scatter.pdf_type = GPUScatterPdfType::GGX;

            scatter.ggx_pdf.normal_x = 0.0f;
            scatter.ggx_pdf.normal_y = -1.0f;
            scatter.ggx_pdf.normal_z = 0.0f;

            scatter.ggx_pdf.view_x = 0.0f;
            scatter.ggx_pdf.view_y = -1.0f;
            scatter.ggx_pdf.view_z = 0.0f;

            scatter.ggx_pdf.tangent_x = 1.0f;
            scatter.ggx_pdf.tangent_y = 0.0f;
            scatter.ggx_pdf.tangent_z = 0.0f;

            scatter.ggx_pdf.bitangent_x = 0.0f;
            scatter.ggx_pdf.bitangent_y = 0.0f;
            scatter.ggx_pdf.bitangent_z = 1.0f;

            scatter.ggx_pdf.roughness = 0.5f;
        }

        GPUMixturePdf mixture{};

        unsigned int seed =
            12345u + index * 7919u + type_index * 104729u;

        float dx = 0.0f;
        float dy = 0.0f;
        float dz = 0.0f;

        bool generated = gpu_mixture_pdf_generate(
            mixture,
            primitives,
            light_indices,
            2,
            hittable_pdf,
            scatter,
            seed,
            dx,
            dy,
            dz
        );

        if (!generated) {
            atomicAdd(&results[base], 1);
        }
        else {
            if (mixture.sample_hittable) {
                atomicAdd(&results[base + 1], 1);
            }
            else {
                atomicAdd(&results[base + 2], 1);
            }

            float length_squared =
                dx * dx + dy * dy + dz * dz;

            float direction_length = sqrtf(length_squared);

            bool invalid_direction =
                !isfinite(dx) ||
                !isfinite(dy) ||
                !isfinite(dz) ||
                !isfinite(direction_length);

            if (!invalid_direction && !mixture.sample_hittable) {
                invalid_direction =
                    fabsf(direction_length - 1.0f) > 1e-4f;
            }

            if (invalid_direction) {
                atomicAdd(&results[base + 3], 1);
            }
        }

        // Evaluate the mixture PDF using a known direction.
        float test_dx = 0.0f;
        float test_dy = -1.0f;
        float test_dz = 2.0f;

        float hittable_value = gpu_hittable_pdf_value(
            primitives,
            light_indices,
            2,
            hittable_pdf,
            test_dx,
            test_dy,
            test_dz
        );

        float material_value = gpu_material_pdf_value(
            scatter,
            test_dx,
            test_dy,
            test_dz
        );

        float mixture_value = gpu_mixture_pdf_value(
            mixture,
            primitives,
            light_indices,
            2,
            hittable_pdf,
            scatter,
            test_dx,
            test_dy,
            test_dz
        );

        float expected_value =
            0.5f * hittable_value +
            0.5f * material_value;

        if (
            !isfinite(hittable_value) ||
            !isfinite(material_value) ||
            !isfinite(mixture_value) ||
            fabsf(mixture_value - expected_value) > 1e-5f
        ) {
            atomicAdd(&results[base + 4], 1);
        }

        if (index == 0) {
            int value_base = type_index * 3;

            pdf_values[value_base] = hittable_value;
            pdf_values[value_base + 1] = material_value;
            pdf_values[value_base + 2] = mixture_value;
        }
    }
}

int main() {
    constexpr int sample_count = 10000;

    int host_results[7] = {};
    float host_mixture_values[3] = {};

    int host_type_results[10] = {};
    float host_type_pdf_values[6] = {};

    int* device_results = nullptr;
    float* device_mixture_values = nullptr;
    int* device_type_results = nullptr;
    float* device_type_pdf_values = nullptr;

    cudaError_t error = cudaMalloc(
        &device_results,
        7 * sizeof(int)
    );

    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << '\n';
        return 1;
    }

    error = cudaMalloc(
        &device_mixture_values,
        3 * sizeof(float)
    );

    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << '\n';
        cudaFree(device_results);
        return 1;
    }

    error = cudaMalloc(
        &device_type_results,
        10 * sizeof(int)
    );

    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << '\n';
        cudaFree(device_results);
        cudaFree(device_mixture_values);
        return 1;
    }

    error = cudaMalloc(
        &device_type_pdf_values,
        6 * sizeof(float)
    );

    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << '\n';
        cudaFree(device_results);
        cudaFree(device_mixture_values);
        cudaFree(device_type_results);
        return 1;
    }

    cudaMemset(device_results, 0, 7 * sizeof(int));
    cudaMemset(device_type_results, 0, 10 * sizeof(int));

    test_gpu_mixture_pdf<<<sample_count, 1>>>(
        device_results,
        device_mixture_values
    );

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        std::cerr << "Mixture kernel failed: "
                  << cudaGetErrorString(error) << '\n';

        cudaFree(device_results);
        cudaFree(device_mixture_values);
        cudaFree(device_type_results);
        cudaFree(device_type_pdf_values);
        return 1;
    }

    test_gpu_mixture_material_types<<<sample_count, 1>>>(
        device_type_results,
        device_type_pdf_values
    );

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        std::cerr << "Material PDF kernel failed: "
                  << cudaGetErrorString(error) << '\n';

        cudaFree(device_results);
        cudaFree(device_mixture_values);
        cudaFree(device_type_results);
        cudaFree(device_type_pdf_values);
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
        cudaFree(device_mixture_values);
        cudaFree(device_type_results);
        cudaFree(device_type_pdf_values);
        return 1;
    }

    error = cudaMemcpy(
        host_mixture_values,
        device_mixture_values,
        3 * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << '\n';

        cudaFree(device_results);
        cudaFree(device_mixture_values);
        cudaFree(device_type_results);
        cudaFree(device_type_pdf_values);
        return 1;
    }

    error = cudaMemcpy(
        host_type_results,
        device_type_results,
        10 * sizeof(int),
        cudaMemcpyDeviceToHost
    );

    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << '\n';

        cudaFree(device_results);
        cudaFree(device_mixture_values);
        cudaFree(device_type_results);
        cudaFree(device_type_pdf_values);
        return 1;
    }

    error = cudaMemcpy(
        host_type_pdf_values,
        device_type_pdf_values,
        6 * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << '\n';

        cudaFree(device_results);
        cudaFree(device_mixture_values);
        cudaFree(device_type_results);
        cudaFree(device_type_pdf_values);
        return 1;
    }

    cudaFree(device_results);
    cudaFree(device_mixture_values);
    cudaFree(device_type_results);
    cudaFree(device_type_pdf_values);

    bool generation_pass = host_results[0] == 0;
    bool hittable_pass = host_results[1] > 0;
    bool cosine_pass = host_results[2] > 0;
    bool direction_pass = host_results[3] == 0;
    bool light_bounds_pass = host_results[4] == 0;
    bool mixture_value_pass = host_results[5] == 0;
    bool empty_lights_pass = host_results[6] == 0;

    std::cout << "Mixture generation: "
              << (generation_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "Hittable branch: "
              << (hittable_pass ? "PASS" : "FAIL")
              << " (" << host_results[1] << ")\n";

    std::cout << "Cosine branch: "
              << (cosine_pass ? "PASS" : "FAIL")
              << " (" << host_results[2] << ")\n";

    std::cout << "Generated directions: "
              << (direction_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "Light sampling boundaries: "
              << (light_bounds_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "Mixture PDF evaluation: "
              << (mixture_value_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "Empty light handling: "
              << (empty_lights_pass ? "PASS" : "FAIL") << '\n';

    std::cout << "\nPDF values:\n";
    std::cout << "Hittable: " << host_mixture_values[0] << '\n';
    std::cout << "Cosine: " << host_mixture_values[1] << '\n';
    std::cout << "Mixture: " << host_mixture_values[2] << '\n';

    std::cout << "\nAdditional material PDF tests:\n";

    const char* type_names[2] = {
        "Sphere",
        "GGX"
    };

    bool additional_tests_pass = true;

    for (int i = 0; i < 2; i++) {
        int base = i * 5;
        int value_base = i * 3;

        bool type_generation_pass =
            host_type_results[base] == 0;

        bool type_hittable_pass =
            host_type_results[base + 1] > 0;

        bool type_material_pass =
            host_type_results[base + 2] > 0;

        bool type_direction_pass =
            host_type_results[base + 3] == 0;

        bool type_pdf_pass =
            host_type_results[base + 4] == 0;

        bool type_pass =
            type_generation_pass &&
            type_hittable_pass &&
            type_material_pass &&
            type_direction_pass &&
            type_pdf_pass;

        additional_tests_pass &= type_pass;

        std::cout << "\n" << type_names[i] << " PDF:\n";

        std::cout << "  Generation: "
                  << (type_generation_pass ? "PASS" : "FAIL") << '\n';

        std::cout << "  Hittable branch: "
                  << (type_hittable_pass ? "PASS" : "FAIL")
                  << " (" << host_type_results[base + 1] << ")\n";

        std::cout << "  Material branch: "
                  << (type_material_pass ? "PASS" : "FAIL")
                  << " (" << host_type_results[base + 2] << ")\n";

        std::cout << "  Generated directions: "
                  << (type_direction_pass ? "PASS" : "FAIL") << '\n';

        std::cout << "  Mixture PDF evaluation: "
                  << (type_pdf_pass ? "PASS" : "FAIL") << '\n';

        std::cout << "  Hittable: "
                  << host_type_pdf_values[value_base] << '\n';

        std::cout << "  Material: "
                  << host_type_pdf_values[value_base + 1] << '\n';

        std::cout << "  Mixture: "
                  << host_type_pdf_values[value_base + 2] << '\n';
    }

    bool all_pass =
        generation_pass &&
        hittable_pass &&
        cosine_pass &&
        direction_pass &&
        light_bounds_pass &&
        mixture_value_pass &&
        empty_lights_pass &&
        additional_tests_pass;

    std::cout << "\nGPU MixturePdf tests: "
              << (all_pass ? "PASS" : "FAIL") << '\n';

    return all_pass ? 0 : 1;
}