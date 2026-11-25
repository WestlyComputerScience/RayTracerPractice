
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common_constants.h"
#include "vec3.h"
#include "interval.h"
#include "ray.h"
#include "aabb.h"
#include "material.h"
#include "quad.h"
#include "hittable_list.h"
#include "camera.h"

#include "gpu_scene_builder.h"
#include "gpu_scene.cuh"
#include "gpu_camera_builder.h"
#include "gpu_render.cuh"
#include "gpu_traversal.cuh"
#include "gpu_intersection.cuh"

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            cudaGetErrorString(error)
        );
    }
}

struct GPUHitComparison {
    int bvh_hit;
    int linear_hit;

    int bvh_primitive_index;
    int linear_primitive_index;

    float bvh_t;
    float linear_t;
};

__global__ void compare_bvh_and_linear_kernel(
    const GPUBvhNode* nodes,
    const GPUPrimitive* primitives,
    int node_count,
    int primitive_count,
    const GPURay* rays,
    GPUHitComparison* results,
    int ray_count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= ray_count) {
        return;
    }

    const GPURay& ray = rays[index];

    // --------------------------------------------------
    // 1. Intersect using the BVH.
    // --------------------------------------------------

    GPUTraversalResult bvh_result = gpu_bvh_hit(
        nodes,
        primitives,
        node_count,
        primitive_count,
        ray,
        0.001f,
        1e30f
    );

    // --------------------------------------------------
    // 2. Intersect by scanning every primitive.
    // --------------------------------------------------

    int linear_hit = 0;
    int linear_primitive_index = -1;

    float linear_t = 1e30f;

    for (int i = 0; i < primitive_count; i++) {
        const GPUPrimitive& primitive = primitives[i];

        float hit_t;
        float hit_x;
        float hit_y;
        float hit_z;

        bool primitive_hit = false;

        if (primitive.type == GPUPrimitiveType::Quad) {

            primitive_hit = gpu_quad_hit(
                primitive,
                ray,
                0.001f,
                linear_t,
                hit_t,
                hit_x,
                hit_y,
                hit_z
            );

        } else if (primitive.type == GPUPrimitiveType::Sphere) {

            float normal_x;
            float normal_y;
            float normal_z;

            primitive_hit = gpu_sphere_hit(
                primitive,
                ray,
                0.001f,
                linear_t,
                hit_t,
                hit_x,
                hit_y,
                hit_z,
                normal_x,
                normal_y,
                normal_z
            );
        }

        if (primitive_hit) {
            linear_hit = 1;
            linear_t = hit_t;
            linear_primitive_index = i;
        }
    }

    // --------------------------------------------------
    // 3. Store both results.
    // --------------------------------------------------

    GPUHitComparison comparison{};

    comparison.bvh_hit = bvh_result.hit;
    comparison.linear_hit = linear_hit;

    comparison.bvh_primitive_index =
        bvh_result.primitive_index;

    comparison.linear_primitive_index =
        linear_primitive_index;

    comparison.bvh_t = bvh_result.t;
    comparison.linear_t = linear_t;

    results[index] = comparison;
}

void render_gpu_scene(
    const GPUCamera& camera,
    const GPUDeviceScene& scene,
    GPUColor* device_pixels,
    int pixel_count
) {
    constexpr int threads_per_block = 256;

    const int blocks =
        (pixel_count + threads_per_block - 1) /
        threads_per_block;

    gpu_render_path_tracing<<<blocks, threads_per_block>>>(
        camera,
        scene.bvh_nodes,
        scene.primitives,
        scene.materials,
        scene.light_indices,
        static_cast<int>(scene.bvh_node_count),
        static_cast<int>(scene.primitive_count),
        static_cast<int>(scene.material_count),
        static_cast<int>(scene.light_count),
        device_pixels
    );

    check_cuda(
        cudaGetLastError(),
        "Launching GPU path-tracing kernel"
    );

    check_cuda(
        cudaDeviceSynchronize(),
        "Executing GPU path-tracing kernel"
    );
}

void compare_gpu_bvh_to_linear(
    const GPUDeviceScene& scene
) {
    constexpr int grid_width = 64;
    constexpr int grid_height = 64;

    constexpr int ray_count =
        grid_width * grid_height;

    constexpr int threads_per_block = 256;

    std::vector<GPURay> host_rays;
    host_rays.reserve(ray_count);

    // Use the same origin for every ray.
    constexpr float origin_x = 278.0f;
    constexpr float origin_y = 278.0f;
    constexpr float origin_z = -800.0f;

    // Aim rays at a grid of points on the back wall.
    for (int y = 0; y < grid_height; y++) {
        for (int x = 0; x < grid_width; x++) {
            float target_x =
                (static_cast<float>(x) + 0.5f) *
                (555.0f / grid_width);

            float target_y =
                (static_cast<float>(y) + 0.5f) *
                (555.0f / grid_height);

            host_rays.push_back(GPURay{
                origin_x,
                origin_y,
                origin_z,

                target_x - origin_x,
                target_y - origin_y,
                555.0f - origin_z
            });
        }
    }

    GPURay* device_rays = nullptr;
    GPUHitComparison* device_results = nullptr;

    check_cuda(
        cudaMalloc(
            reinterpret_cast<void**>(&device_rays),
            ray_count * sizeof(GPURay)
        ),
        "Allocating diagnostic rays"
    );

    check_cuda(
        cudaMalloc(
            reinterpret_cast<void**>(&device_results),
            ray_count * sizeof(GPUHitComparison)
        ),
        "Allocating diagnostic results"
    );

    check_cuda(
        cudaMemcpy(
            device_rays,
            host_rays.data(),
            ray_count * sizeof(GPURay),
            cudaMemcpyHostToDevice
        ),
        "Copying diagnostic rays"
    );

    int blocks =
        (ray_count + threads_per_block - 1) /
        threads_per_block;

    compare_bvh_and_linear_kernel<<<
        blocks,
        threads_per_block
    >>>(
        scene.bvh_nodes,
        scene.primitives,
        static_cast<int>(scene.bvh_node_count),
        static_cast<int>(scene.primitive_count),
        device_rays,
        device_results,
        ray_count
    );

    check_cuda(
        cudaGetLastError(),
        "Launching BVH comparison kernel"
    );

    check_cuda(
        cudaDeviceSynchronize(),
        "Executing BVH comparison kernel"
    );

    std::vector<GPUHitComparison> results(ray_count);

    check_cuda(
        cudaMemcpy(
            results.data(),
            device_results,
            ray_count * sizeof(GPUHitComparison),
            cudaMemcpyDeviceToHost
        ),
        "Copying BVH comparison results"
    );

    int hit_mismatches = 0;
    int distance_mismatches = 0;
    int primitive_mismatches = 0;
    int stack_overflows = 0;

    constexpr float distance_tolerance = 1e-4f;

    for (int i = 0; i < ray_count; i++) {
        const GPUHitComparison& result = results[i];

        if (result.bvh_hit != result.linear_hit) {
            hit_mismatches++;
        }

        if (result.bvh_hit && result.linear_hit) {
            if (
                std::fabs(result.bvh_t - result.linear_t) >
                distance_tolerance
            ) {
                distance_mismatches++;
            }

            if (
                result.bvh_primitive_index !=
                result.linear_primitive_index
            ) {
                primitive_mismatches++;
            }
        }

        if (
            !result.bvh_hit &&
            result.bvh_primitive_index != -1
        ) {
            stack_overflows++;
        }

        if (
            hit_mismatches <= 5 &&
            (
                result.bvh_hit != result.linear_hit ||
                (
                    result.bvh_hit &&
                    result.linear_hit &&
                    std::fabs(result.bvh_t - result.linear_t) >
                        distance_tolerance
                )
            )
        ) {
            std::cout
                << "\nRay " << i
                << "\n  BVH hit: " << result.bvh_hit
                << "\n  Linear hit: " << result.linear_hit
                << "\n  BVH t: " << result.bvh_t
                << "\n  Linear t: " << result.linear_t
                << "\n  BVH primitive: "
                << result.bvh_primitive_index
                << "\n  Linear primitive: "
                << result.linear_primitive_index
                << '\n';
        }
    }

    std::cout
        << "\n========================================\n"
        << "GPU BVH / LINEAR INTERSECTION TEST\n"
        << "========================================\n"
        << "Rays tested: " << ray_count << '\n'
        << "Hit/miss mismatches: " << hit_mismatches << '\n'
        << "Distance mismatches: " << distance_mismatches << '\n'
        << "Primitive mismatches: " << primitive_mismatches << '\n'
        << "BVH stack overflow indicators: " << stack_overflows << '\n';

    check_cuda(
        cudaFree(device_rays),
        "Freeing diagnostic rays"
    );

    check_cuda(
        cudaFree(device_results),
        "Freeing diagnostic results"
    );
}

int main() {
    GPUDeviceScene device_scene{};
    GPUColor* device_pixels = nullptr;
    float* cpu_pixels = nullptr;

    try {
        constexpr int image_width = 256;
        constexpr int image_height = 256;
        constexpr int pixel_count = image_width * image_height;

        constexpr int samples_per_pixel = 1024;
        constexpr int max_bounces = 5;

        // --------------------------------------------------
        // 1. Construct the Cornell scene.
        // --------------------------------------------------

        auto red = make_shared<Lambertian>(
            Color(0.65, 0.05, 0.05)
        );

        auto white = make_shared<Lambertian>(
            Color(0.73, 0.73, 0.73)
        );

        auto green = make_shared<Lambertian>(
            Color(0.12, 0.45, 0.15)
        );

        auto light = make_shared<DiffuseLight>(
            Color(15, 15, 15)
        );

        HittableList world;
        HittableList lights;

        world.add(make_shared<Quad>(
            Point3(555, 0, 0),
            Vec3(0, 555, 0),
            Vec3(0, 0, 555),
            green
        ));

        world.add(make_shared<Quad>(
            Point3(0, 0, 0),
            Vec3(0, 555, 0),
            Vec3(0, 0, 555),
            red
        ));

        world.add(make_shared<Quad>(
            Point3(343, 554, 332),
            Vec3(-130, 0, 0),
            Vec3(0, 0, -105),
            light
        ));

        world.add(make_shared<Quad>(
            Point3(0, 0, 0),
            Vec3(555, 0, 0),
            Vec3(0, 0, 555),
            white
        ));

        world.add(make_shared<Quad>(
            Point3(555, 555, 555),
            Vec3(-555, 0, 0),
            Vec3(0, 0, -555),
            white
        ));

        world.add(make_shared<Quad>(
            Point3(0, 0, 555),
            Vec3(555, 0, 0),
            Vec3(0, 555, 0),
            white
        ));

        // The CPU light list contains the same light geometry.
        // --------------------------------------------------
        // Add GPU sphere-intersection test objects.
        // --------------------------------------------------

        auto blue = make_shared<Lambertian>(
            Color(0.05, 0.15, 0.65)
        );

        auto yellow = make_shared<Lambertian>(
            Color(0.65, 0.45, 0.05)
        );

        world.add(make_shared<Sphere>(
            Point3(180, 100, 300),
            100,
            blue
        ));

        world.add(make_shared<Sphere>(
            Point3(370, 100, 400),
            100,
            yellow
        ));

        // Its material is intentionally null.
        shared_ptr<Material> empty_material;

        lights.add(make_shared<Quad>(
            Point3(343, 554, 332),
            Vec3(-130, 0, 0),
            Vec3(0, 0, -105),
            empty_material
        ));

        // --------------------------------------------------
        // 2. Configure the CPU camera.
        // --------------------------------------------------

        Camera camera;

        camera.config.aspect_ratio = 1.0;
        camera.config.image_width = image_width;
        camera.config.samples_per_pixel = samples_per_pixel;
        camera.config.max_ray_bounces = max_bounces;

        camera.config.background = Color(0, 0, 0);

        camera.config.vfov = 40;

        camera.config.lookfrom = Point3(278, 278, -800);
        camera.config.lookat = Point3(278, 278, 0);
        camera.config.vup = Vec3(0, 1, 0);

        camera.config.defocus_angle = 0;

        // Initialize once so both renderers use the same
        // initialized camera geometry.
        camera.initialize();

        // --------------------------------------------------
        // 3. Build and upload the GPU scene.
        // --------------------------------------------------

        GPUScene host_scene =
            build_gpu_scene_from_list(world);

        if (host_scene.primitives.size() != 8) {
            throw std::runtime_error(
                "Expected exactly eight GPU primitives: "
                "six Quads and two Spheres."
            );
        }

        if (host_scene.light_indices.size() != 1) {
            throw std::runtime_error(
                "Expected exactly one GPU light."
            );
        }

        device_scene = upload_gpu_scene(host_scene);

        const GPUBvhNode& root = host_scene.bvh_nodes[0];

        std::cout << "\nBVH ROOT BOUNDS:\n"
                << "Min: "
                << root.min_x << ", "
                << root.min_y << ", "
                << root.min_z << '\n'
                << "Max: "
                << root.max_x << ", "
                << root.max_y << ", "
                << root.max_z << '\n';

        std::cout << "\nGPU PRIMITIVE BOUNDS:\n";

        for (size_t i = 0; i < host_scene.primitives.size(); i++) {
            const GPUPrimitive& p = host_scene.primitives[i];

            std::cout << "Primitive " << i << ": "
                    << "min=("
                    << p.min_x << ", "
                    << p.min_y << ", "
                    << p.min_z << "), max=("
                    << p.max_x << ", "
                    << p.max_y << ", "
                    << p.max_z << ")\n";
        }

        std::cout << "\nGPU BVH NODES:\n";

        for (size_t i = 0; i < host_scene.bvh_nodes.size(); i++) {
            const GPUBvhNode& node = host_scene.bvh_nodes[i];

            std::cout << "Node " << i
                    << ": min=("
                    << node.min_x << ", "
                    << node.min_y << ", "
                    << node.min_z << "), max=("
                    << node.max_x << ", "
                    << node.max_y << ", "
                    << node.max_z << ")"
                    << ", left=" << node.left_child
                    << ", right=" << node.right_child
                    << ", primitive_start=" << node.primitive_start
                    << ", primitive_count=" << node.primitive_count
                    << '\n';
        }

        std::cout << "\nTesting GPU BVH against linear intersection...\n";

        compare_gpu_bvh_to_linear(device_scene);

        GPUCamera gpu_camera =
            build_gpu_camera(camera);

        // --------------------------------------------------
        // 4. Render the CPU reference image.
        // --------------------------------------------------

        std::cout << "\nRendering CPU reference...\n";

        camera.render_initialized(world, lights);

        int cpu_width = 0;
        int cpu_height = 0;
        int cpu_channels = 0;

        cpu_pixels = stbi_loadf(
            "cool.hdr",
            &cpu_width,
            &cpu_height,
            &cpu_channels,
            3
        );

        if (cpu_pixels == nullptr) {
            throw std::runtime_error(
                "Could not load CPU HDR output: cool.hdr"
            );
        }

        if (
            cpu_width != image_width ||
            cpu_height != image_height
        ) {
            throw std::runtime_error(
                "CPU HDR dimensions do not match the test dimensions."
            );
        }

        // --------------------------------------------------
        // 5. Render the GPU image.
        // --------------------------------------------------

        std::cout << "\nRendering GPU comparison image...\n";

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_pixels),
                pixel_count * sizeof(GPUColor)
            ),
            "Allocating GPU pixels"
        );

        std::vector<GPUColor> gpu_pixels(pixel_count);

        render_gpu_scene(
            gpu_camera,
            device_scene,
            device_pixels,
            pixel_count
        );

        check_cuda(
            cudaMemcpy(
                gpu_pixels.data(),
                device_pixels,
                pixel_count * sizeof(GPUColor),
                cudaMemcpyDeviceToHost
            ),
            "Copying GPU pixels"
        );

        // --------------------------------------------------
        // 6. Compare CPU and GPU floating-point results.
        // --------------------------------------------------

        double cpu_sum[3] = {0.0, 0.0, 0.0};
        double gpu_sum[3] = {0.0, 0.0, 0.0};

        double absolute_error_sum = 0.0;
        double squared_error_sum = 0.0;

        double max_absolute_error = 0.0;

        double cpu_luminance_sum = 0.0;
        double gpu_luminance_sum = 0.0;

        int invalid_pixels = 0;

        for (int i = 0; i < pixel_count; i++) {
            const float* cpu = &cpu_pixels[i * 3];
            const GPUColor& gpu = gpu_pixels[i];

            const float gpu_channels[3] = {
                gpu.r,
                gpu.g,
                gpu.b
            };

            bool valid = true;

            for (int channel = 0; channel < 3; channel++) {
                if (
                    !std::isfinite(cpu[channel]) ||
                    !std::isfinite(gpu_channels[channel])
                ) {
                    valid = false;
                    break;
                }
            }

            if (!valid) {
                invalid_pixels++;
                continue;
            }

            for (int channel = 0; channel < 3; channel++) {
                const double difference =
                    std::fabs(
                        static_cast<double>(cpu[channel]) -
                        gpu_channels[channel]
                    );

                cpu_sum[channel] += cpu[channel];
                gpu_sum[channel] += gpu_channels[channel];

                absolute_error_sum += difference;
                squared_error_sum += difference * difference;

                max_absolute_error =
                    std::max(max_absolute_error, difference);
            }

            cpu_luminance_sum +=
                0.2126 * cpu[0] +
                0.7152 * cpu[1] +
                0.0722 * cpu[2];

            gpu_luminance_sum +=
                0.2126 * gpu.r +
                0.7152 * gpu.g +
                0.0722 * gpu.b;
        }

        const int valid_channel_count =
            (pixel_count - invalid_pixels) * 3;

        if (valid_channel_count == 0) {
            throw std::runtime_error(
                "No valid pixels available for comparison."
            );
        }

        const double mean_absolute_error =
            absolute_error_sum / valid_channel_count;

        const double root_mean_squared_error =
            std::sqrt(
                squared_error_sum / valid_channel_count
            );

        std::cout << std::fixed << std::setprecision(6);

        std::cout << "\n========================================\n";
        std::cout << "CPU / GPU PATH-TRACING COMPARISON\n";
        std::cout << "========================================\n";

        std::cout
            << "Resolution: "
            << image_width << " x " << image_height << '\n';

        std::cout
            << "Samples per pixel: "
            << samples_per_pixel << '\n';

        std::cout
            << "Maximum bounces: "
            << max_bounces << '\n';

        std::cout << "\nAverage RGB:\n";

        std::cout
            << "CPU: "
            << cpu_sum[0] / pixel_count << ", "
            << cpu_sum[1] / pixel_count << ", "
            << cpu_sum[2] / pixel_count << '\n';

        std::cout
            << "GPU: "
            << gpu_sum[0] / pixel_count << ", "
            << gpu_sum[1] / pixel_count << ", "
            << gpu_sum[2] / pixel_count << '\n';

        std::cout << "\nAverage luminance:\n";

        std::cout
            << "CPU: "
            << cpu_luminance_sum / pixel_count << '\n';

        std::cout
            << "GPU: "
            << gpu_luminance_sum / pixel_count << '\n';

        std::cout << "\nImage-wide error:\n";

        std::cout
            << "Mean absolute error: "
            << mean_absolute_error << '\n';

        std::cout
            << "Root mean squared error: "
            << root_mean_squared_error << '\n';

        std::cout
            << "Maximum absolute error: "
            << max_absolute_error << '\n';

        std::cout
            << "Invalid pixels: "
            << invalid_pixels << '\n';

        std::cout
            << "Valid pixels: "
            << pixel_count - invalid_pixels
            << " / " << pixel_count << '\n';

        std::cout
            << "\nComparison completed. "
            << "CPU and GPU use independent random streams.\n";

        // --------------------------------------------------
        // 7. Release resources.
        // --------------------------------------------------

        stbi_image_free(cpu_pixels);
        cpu_pixels = nullptr;

        check_cuda(
            cudaFree(device_pixels),
            "Freeing GPU pixels"
        );

        device_pixels = nullptr;

        free_gpu_scene(device_scene);

        return invalid_pixels == 0 ? 0 : 1;
    }
    catch (const std::exception& error) {
        if (cpu_pixels != nullptr) {
            stbi_image_free(cpu_pixels);
        }

        if (device_pixels != nullptr) {
            cudaFree(device_pixels);
        }

        free_gpu_scene(device_scene);

        std::cerr << "\nERROR: " << error.what() << '\n';

        return 1;
    }
}