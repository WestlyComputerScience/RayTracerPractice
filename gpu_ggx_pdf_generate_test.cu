
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "common_constants.h"
#include "pdf.h"
#include "gpu_pdf.cuh"

constexpr int TEST_COUNT = 6;
constexpr float TOLERANCE = 0.00001f;

struct GGXGenerateTestCase {
    GPUGGXPdf pdf;
    unsigned int seed;
};

struct GGXGenerateResult {
    float dx;
    float dy;
    float dz;
    unsigned int final_seed;
};

unsigned int cpu_random_uint(unsigned int& seed) {
    if (seed == 0u) {
        seed = 0x6D2B79F5u;
    }

    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;

    return seed;
}

float cpu_random_float(unsigned int& seed) {
    return static_cast<float>(
        cpu_random_uint(seed) >> 8
    ) * (1.0f / 16777216.0f);
}

__global__ void test_gpu_ggx_pdf_generate(
    const GGXGenerateTestCase* cases,
    GGXGenerateResult* results,
    int count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count) {
        return;
    }

    unsigned int seed = cases[index].seed;

    float dx = 0.0f;
    float dy = 0.0f;
    float dz = 0.0f;

    gpu_ggx_pdf_generate(
        cases[index].pdf,
        seed,
        dx,
        dy,
        dz
    );

    results[index].dx = dx;
    results[index].dy = dy;
    results[index].dz = dz;
    results[index].final_seed = seed;
}

int main() {
    const Vec3 normals[TEST_COUNT] = {
        Vec3(0, 0, 1),
        Vec3(0, 1, 1),
        Vec3(1, 0, 1),
        Vec3(0, 1, 2),
        Vec3(1, 2, 1),
        Vec3(1, 1, 1)
    };

    const Vec3 views[TEST_COUNT] = {
        Vec3(0.3, 0.2, 1.0),
        Vec3(0.2, 0.5, 1.0),
        Vec3(1.0, 0.3, 1.0),
        Vec3(0.4, 0.8, 1.0),
        Vec3(0.5, 1.0, 0.7),
        Vec3(0.8, 0.4, 1.0)
    };

    const float roughness[TEST_COUNT] = {
        0.2f, 0.35f, 0.5f, 0.7f, 0.9f, 1.0f
    };

    const unsigned int initial_seeds[TEST_COUNT] = {
        12345u,
        98765u,
        0x12345678u,
        0xABCDEF01u,
        0x6D2B79F5u,
        0x87654321u
    };

    GGXGenerateTestCase host_cases[TEST_COUNT]{};

    for (int i = 0; i < TEST_COUNT; i++) {
        Vec3 normal = unit_vector(normals[i]);
        Vec3 view = unit_vector(views[i]);

        Vec3 tangent = unit_vector(
            cross(
                std::fabs(normal.x()) > 0.9
                    ? Vec3(0, 1, 0)
                    : Vec3(1, 0, 0),
                normal
            )
        );

        Vec3 bitangent = cross(normal, tangent);

        GPUGGXPdf& pdf = host_cases[i].pdf;

        pdf.normal_x = static_cast<float>(normal.x());
        pdf.normal_y = static_cast<float>(normal.y());
        pdf.normal_z = static_cast<float>(normal.z());

        pdf.view_x = static_cast<float>(view.x());
        pdf.view_y = static_cast<float>(view.y());
        pdf.view_z = static_cast<float>(view.z());

        pdf.tangent_x = static_cast<float>(tangent.x());
        pdf.tangent_y = static_cast<float>(tangent.y());
        pdf.tangent_z = static_cast<float>(tangent.z());

        pdf.bitangent_x = static_cast<float>(bitangent.x());
        pdf.bitangent_y = static_cast<float>(bitangent.y());
        pdf.bitangent_z = static_cast<float>(bitangent.z());

        pdf.roughness = roughness[i];

        host_cases[i].seed = initial_seeds[i];
    }

    GGXGenerateTestCase* device_cases = nullptr;
    GGXGenerateResult* device_results = nullptr;

    cudaError_t error = cudaMalloc(
        &device_cases,
        sizeof(host_cases)
    );

    if (error != cudaSuccess) {
        std::cerr << "cudaMalloc cases failed: "
                  << cudaGetErrorString(error) << '\n';
        return 1;
    }

    error = cudaMalloc(
        &device_results,
        sizeof(GGXGenerateResult) * TEST_COUNT
    );

    if (error != cudaSuccess) {
        std::cerr << "cudaMalloc results failed: "
                  << cudaGetErrorString(error) << '\n';
        cudaFree(device_cases);
        return 1;
    }

    error = cudaMemcpy(
        device_cases,
        host_cases,
        sizeof(host_cases),
        cudaMemcpyHostToDevice
    );

    if (error != cudaSuccess) {
        std::cerr << "cudaMemcpy cases failed: "
                  << cudaGetErrorString(error) << '\n';
        cudaFree(device_cases);
        cudaFree(device_results);
        return 1;
    }

    test_gpu_ggx_pdf_generate<<<1, TEST_COUNT>>>(
        device_cases,
        device_results,
        TEST_COUNT
    );

    error = cudaGetLastError();

    if (error == cudaSuccess) {
        error = cudaDeviceSynchronize();
    }

    if (error != cudaSuccess) {
        std::cerr << "CUDA execution failed: "
                  << cudaGetErrorString(error) << '\n';
        cudaFree(device_cases);
        cudaFree(device_results);
        return 1;
    }

    GGXGenerateResult gpu_results[TEST_COUNT]{};

    error = cudaMemcpy(
        gpu_results,
        device_results,
        sizeof(gpu_results),
        cudaMemcpyDeviceToHost
    );

    cudaFree(device_cases);
    cudaFree(device_results);

    if (error != cudaSuccess) {
        std::cerr << "cudaMemcpy results failed: "
                  << cudaGetErrorString(error) << '\n';
        return 1;
    }

    bool all_passed = true;

    std::cout << "Production GGX PDF generator test:\n\n";

    for (int i = 0; i < TEST_COUNT; i++) {
        const GPUGGXPdf& pdf = host_cases[i].pdf;

        Vec3 normal(
            pdf.normal_x,
            pdf.normal_y,
            pdf.normal_z
        );

        Vec3 view(
            pdf.view_x,
            pdf.view_y,
            pdf.view_z
        );

        Vec3 tangent(
            pdf.tangent_x,
            pdf.tangent_y,
            pdf.tangent_z
        );

        Vec3 bitangent(
            pdf.bitangent_x,
            pdf.bitangent_y,
            pdf.bitangent_z
        );

        // Reproduce the production GPU RNG sequence.
        unsigned int expected_seed = initial_seeds[i];

        float r1 = cpu_random_float(expected_seed);
        float r2 = cpu_random_float(expected_seed);

        // Reconstruct the expected CPU GGX sample.
        Vec3 view_local(
            dot(view, tangent),
            dot(view, bitangent),
            dot(view, normal)
        );

        Vec3 h_local = sample_ggx_vndf(
            view_local,
            pdf.roughness,
            r1,
            r2
        );

        Vec3 h = unit_vector(
            tangent * h_local.x() +
            bitangent * h_local.y() +
            normal * h_local.z()
        );

        Vec3 expected_direction = reflect(-view, h);

        const GGXGenerateResult& gpu = gpu_results[i];

        float dx = std::fabs(
            static_cast<float>(expected_direction.x()) - gpu.dx
        );

        float dy = std::fabs(
            static_cast<float>(expected_direction.y()) - gpu.dy
        );

        float dz = std::fabs(
            static_cast<float>(expected_direction.z()) - gpu.dz
        );

        float max_difference = fmaxf(
            dx,
            fmaxf(dy, dz)
        );

        float gpu_length = sqrtf(
            gpu.dx * gpu.dx +
            gpu.dy * gpu.dy +
            gpu.dz * gpu.dz
        );

        bool finite =
            std::isfinite(gpu.dx) &&
            std::isfinite(gpu.dy) &&
            std::isfinite(gpu.dz);

        bool normalized =
            std::fabs(gpu_length - 1.0f) <= TOLERANCE;

        bool seed_passed =
            gpu.final_seed == expected_seed;

        bool direction_passed =
            max_difference <= TOLERANCE;

        bool passed =
            finite &&
            normalized &&
            seed_passed &&
            direction_passed;

        if (!passed) {
            all_passed = false;
        }

        std::cout
            << "Test " << i + 1 << ": "
            << (passed ? "PASS" : "FAIL") << '\n'
            << "  Initial seed: " << initial_seeds[i] << '\n'
            << "  Random values: (" << r1 << ", " << r2 << ")\n"
            << "  Expected final seed: " << expected_seed << '\n'
            << "  GPU final seed: " << gpu.final_seed << '\n'
            << "  Expected direction: ("
            << expected_direction.x() << ", "
            << expected_direction.y() << ", "
            << expected_direction.z() << ")\n"
            << "  GPU direction: ("
            << gpu.dx << ", "
            << gpu.dy << ", "
            << gpu.dz << ")\n"
            << "  Maximum direction difference: "
            << max_difference << '\n'
            << "  GPU direction length: "
            << gpu_length << '\n'
            << "  Seed advancement: "
            << (seed_passed ? "PASS" : "FAIL")
            << "\n\n";
    }

    std::cout
        << "Production GGX PDF generator: "
        << (all_passed ? "PASS" : "FAIL")
        << '\n';

    return all_passed ? 0 : 1;
}