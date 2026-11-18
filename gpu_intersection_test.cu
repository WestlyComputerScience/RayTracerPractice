#include "common_constants.h"
#include "vec3.h"
#include "ray.h"
#include "interval.h"
#include "aabb.h"
#include "gpu_intersection.cuh"

__global__ void test_gpu_aabb(
    GPUBvhNode box,
    const GPURay* rays,
    bool* results,
    int ray_count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= ray_count) return;

    results[index] = gpu_aabb_hit(
        box,
        rays[index],
        0.001f,
        1000.0f
    );
}

int main() {
    GPUBvhNode gpu_box{};

    gpu_box.min_x = 0.0f;
    gpu_box.min_y = 0.0f;
    gpu_box.min_z = 0.0f;

    gpu_box.max_x = 1.0f;
    gpu_box.max_y = 1.0f;
    gpu_box.max_z = 1.0f;

    Aabb cpu_box(
        Point3(0.0, 0.0, 0.0),
        Point3(1.0, 1.0, 1.0)
    );

    std::vector<GPURay> rays = {
        // 0: Hits the box from the negative X direction.
        {-1.0f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f},

        // 1: Misses above the box.
        {-1.0f, 2.0f, 0.5f, 1.0f, 0.0f, 0.0f},

        // 2: Parallel to X and inside the other two slabs.
        {0.5f, -1.0f, 0.5f, 0.0f, 1.0f, 0.0f},

        // 3: Parallel to X and outside the X slab.
        {-1.0f, -1.0f, 0.5f, 0.0f, 1.0f, 0.0f},

        // 4: Hits from the positive Z direction.
        {0.5f, 0.5f, 2.0f, 0.0f, 0.0f, -1.0f},

        // 5: Travels away from the box.
        {-1.0f, 0.5f, 0.5f, -1.0f, 0.0f, 0.0f},

        // 6: Zero direction, starts inside the box.
        {0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 0.0f},

        // 7: Starts on the X boundary and travels inward.
        {0.0f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f}
    };

    const int ray_count = static_cast<int>(rays.size());

    // Allocate GPU memory.
    GPURay* device_rays = nullptr;
    bool* device_results = nullptr;

    cudaMalloc(&device_rays, ray_count * sizeof(GPURay));
    cudaMalloc(&device_results, ray_count * sizeof(bool));

    cudaMemcpy(
        device_rays,
        rays.data(),
        ray_count * sizeof(GPURay),
        cudaMemcpyHostToDevice
    );

    // Run GPU intersection tests.
    test_gpu_aabb<<<1, 32>>>(
        gpu_box,
        device_rays,
        device_results,
        ray_count
    );

    cudaDeviceSynchronize();

    std::vector<unsigned char> gpu_results(ray_count);

    cudaMemcpy(
        gpu_results.data(),
        device_results,
        ray_count * sizeof(bool),
        cudaMemcpyDeviceToHost
    );

    // Run CPU intersection tests and compare.
    int matching_results = 0;

    std::cout << "CPU vs GPU AABB intersection results:\n\n";

    for (int i = 0; i < ray_count; i++) {
        const GPURay& gpu_ray = rays[i];

        Ray cpu_ray(
            Point3(
                gpu_ray.origin_x,
                gpu_ray.origin_y,
                gpu_ray.origin_z
            ),
            Vec3(
                gpu_ray.direction_x,
                gpu_ray.direction_y,
                gpu_ray.direction_z
            )
        );

        bool cpu_hit = cpu_box.hit(
            cpu_ray,
            Interval(0.001, 1000.0)
        );

        bool gpu_hit = gpu_results[i];

        bool matches = cpu_hit == gpu_hit;

        if (matches) {
            matching_results++;
        }

        std::cout
            << "Ray " << i
            << ": CPU = " << (cpu_hit ? "HIT " : "MISS")
            << ", GPU = " << (gpu_hit ? "HIT " : "MISS")
            << ", " << (matches ? "MATCH" : "MISMATCH")
            << '\n';
    }

    std::cout << "\nMatching results: "
              << matching_results
              << " / " << ray_count << '\n';

    // Release GPU memory.
    cudaFree(device_rays);
    cudaFree(device_results);

    return matching_results == ray_count ? 0 : 1;
}