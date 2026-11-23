#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "common_constants.h"
#include "pdf.h"

#include "gpu_vec3.cuh"
#include "gpu_pdf.cuh"

struct GGXTestCase {
    float nx, ny, nz;
    float vx, vy, vz;
    float roughness;
    float r1, r2;
};

struct GGXSampleResult {
    float dx, dy, dz;
    float pdf;
};

__device__ inline GPUVec3 test_gpu_cross(
    const GPUVec3& a,
    const GPUVec3& b
) {
    return GPUVec3{
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

__global__ void test_gpu_ggx_world_space_sampling(
    const GGXTestCase* cases,
    GGXSampleResult* results,
    int count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count)
        return;

    GGXTestCase test = cases[index];

    GPUVec3 normal = gpu_unit_vector(
        GPUVec3{test.nx, test.ny, test.nz}
    );

    GPUVec3 view = gpu_unit_vector(
        GPUVec3{test.vx, test.vy, test.vz}
    );

    // Match the CPU GGXPdf tangent construction.
    GPUVec3 axis = fabsf(normal.x) > 0.9f
        ? GPUVec3{0.0f, 1.0f, 0.0f}
        : GPUVec3{1.0f, 0.0f, 0.0f};

    GPUVec3 tangent = gpu_unit_vector(
        test_gpu_cross(axis, normal)
    );

    GPUVec3 bitangent = test_gpu_cross(normal, tangent);

    // Construct the GPU PDF using the same world-space frame.
    GPUGGXPdf pdf{};

    pdf.normal_x = normal.x;
    pdf.normal_y = normal.y;
    pdf.normal_z = normal.z;

    pdf.view_x = view.x;
    pdf.view_y = view.y;
    pdf.view_z = view.z;

    pdf.tangent_x = tangent.x;
    pdf.tangent_y = tangent.y;
    pdf.tangent_z = tangent.z;

    pdf.bitangent_x = bitangent.x;
    pdf.bitangent_y = bitangent.y;
    pdf.bitangent_z = bitangent.z;

    pdf.roughness = test.roughness;

    // Transform the view direction into tangent space.
    GPUVec3 view_local{
        gpu_dot(view, tangent),
        gpu_dot(view, bitangent),
        gpu_dot(view, normal)
    };

    // Use exactly the same random values as the CPU test.
    GPUVec3 h_local = gpu_sample_ggx_vndf(
        view_local,
        test.roughness,
        test.r1,
        test.r2
    );

    // Transform the sampled microfacet normal into world space.
    GPUVec3 h = gpu_unit_vector(
        GPUVec3{
            tangent.x * h_local.x +
                bitangent.x * h_local.y +
                normal.x * h_local.z,

            tangent.y * h_local.x +
                bitangent.y * h_local.y +
                normal.y * h_local.z,

            tangent.z * h_local.x +
                bitangent.z * h_local.y +
                normal.z * h_local.z
        }
    );

    // Match CPU reflect(-view, h).
    GPUVec3 incoming{
        -view.x,
        -view.y,
        -view.z
    };

    GPUVec3 direction = gpu_reflect(incoming, h);

    results[index].dx = direction.x;
    results[index].dy = direction.y;
    results[index].dz = direction.z;

    results[index].pdf = gpu_ggx_pdf_value(
        pdf,
        direction.x,
        direction.y,
        direction.z
    );
}

int main() {
    constexpr int count = 6;

    GGXTestCase cases[count] = {
        {0, 0, 1,       0.3f, 0.2f, 1.0f, 0.2f, 0.05f, 0.10f},
        {0, 1, 1,       0.2f, 0.5f, 1.0f, 0.35f, 0.20f, 0.30f},
        {1, 0, 1,       1.0f, 0.3f, 1.0f, 0.5f, 0.40f, 0.50f},
        {0, 1, 2,       0.4f, 0.8f, 1.0f, 0.7f, 0.70f, 0.80f},
        {1, 2, 1,       0.5f, 1.0f, 0.7f, 0.9f, 0.95f, 0.95f},
        {1, 1, 1,       0.8f, 0.4f, 1.0f, 1.0f, 0.99f, 0.02f}
    };

    GGXSampleResult gpu_results[count]{};

    GGXTestCase* device_cases = nullptr;
    GGXSampleResult* device_results = nullptr;

    cudaMalloc(&device_cases, sizeof(cases));
    cudaMalloc(&device_results, sizeof(gpu_results));

    cudaMemcpy(
        device_cases,
        cases,
        sizeof(cases),
        cudaMemcpyHostToDevice
    );

    test_gpu_ggx_world_space_sampling<<<1, count>>>(
        device_cases,
        device_results,
        count
    );

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        std::cerr << "Kernel launch failed: "
                  << cudaGetErrorString(error) << '\n';
        return 1;
    }

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        std::cerr << "Kernel execution failed: "
                  << cudaGetErrorString(error) << '\n';
        return 1;
    }

    cudaMemcpy(
        gpu_results,
        device_results,
        sizeof(gpu_results),
        cudaMemcpyDeviceToHost
    );

    cudaFree(device_cases);
    cudaFree(device_results);

    bool all_passed = true;

    std::cout << "GGX world-space sampling CPU/GPU comparison:\n\n";

    for (int i = 0; i < count; ++i) {
        const GGXTestCase& test = cases[i];

        // Reconstruct the CPU GGXPdf.
        Vec3 normal = unit_vector(
            Vec3(test.nx, test.ny, test.nz)
        );

        Vec3 view = unit_vector(
            Vec3(test.vx, test.vy, test.vz)
        );

        GGXPdf cpu_pdf(
            normal,
            view,
            test.roughness
        );

        // Reconstruct the exact CPU tangent frame.
        Vec3 tangent = unit_vector(
            cross(
                std::fabs(normal.x()) > 0.9
                    ? Vec3(0, 1, 0)
                    : Vec3(1, 0, 0),
                normal
            )
        );

        Vec3 bitangent = cross(normal, tangent);

        // Transform the view into tangent space.
        Vec3 view_local(
            dot(view, tangent),
            dot(view, bitangent),
            dot(view, normal)
        );

        // Use the same random values supplied to the GPU.
        Vec3 h_local = sample_ggx_vndf(
            view_local,
            test.roughness,
            test.r1,
            test.r2
        );

        // Transform the sampled microfacet normal into world space.
        Vec3 h = unit_vector(
            tangent * h_local.x() +
            bitangent * h_local.y() +
            normal * h_local.z()
        );

        // Generate the CPU reflected direction.
        Vec3 cpu_direction = reflect(-view, h);

        // Evaluate the CPU GGX PDF.
        real cpu_value = cpu_pdf.value(cpu_direction);

        const GGXSampleResult& gpu = gpu_results[i];

        float dx = std::fabs(
            static_cast<float>(cpu_direction.x()) - gpu.dx
        );

        float dy = std::fabs(
            static_cast<float>(cpu_direction.y()) - gpu.dy
        );

        float dz = std::fabs(
            static_cast<float>(cpu_direction.z()) - gpu.dz
        );

        float direction_difference = fmaxf(
            dx,
            fmaxf(dy, dz)
        );

        float pdf_difference = std::fabs(
            static_cast<float>(cpu_value) - gpu.pdf
        );

        constexpr float direction_tolerance = 0.00001f;
        constexpr float pdf_relative_tolerance = 0.0001f;

        float pdf_tolerance = pdf_relative_tolerance *
            fmaxf(1.0f, std::fabs(static_cast<float>(cpu_value)));

        bool direction_passed =
            direction_difference <= direction_tolerance;

        bool pdf_passed =
            pdf_difference <= pdf_tolerance;

        bool passed = direction_passed && pdf_passed;

        if (!passed)
            all_passed = false;

        std::cout << "Test " << (i + 1) << ": "
                  << (passed ? "PASS" : "FAIL") << '\n';

        std::cout << "  CPU direction: ("
                  << cpu_direction.x() << ", "
                  << cpu_direction.y() << ", "
                  << cpu_direction.z() << ")\n";

        std::cout << "  GPU direction: ("
                  << gpu.dx << ", "
                  << gpu.dy << ", "
                  << gpu.dz << ")\n";

        std::cout << "  Direction difference: "
                  << direction_difference << '\n';

        std::cout << "  CPU PDF: " << cpu_value << '\n';
        std::cout << "  GPU PDF: " << gpu.pdf << '\n';

        std::cout << "  PDF difference: "
                  << pdf_difference << "\n\n";
    }

    std::cout << "GGX world-space sampling CPU/GPU comparison: "
              << (all_passed ? "PASS" : "FAIL")
              << '\n';

    return all_passed ? 0 : 1;
}