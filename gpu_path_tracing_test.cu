
#include <cuda_runtime.h>

#include <cmath>
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

#include "gpu_scene_builder.h"
#include "gpu_scene.cuh"
#include "gpu_camera_builder.h"
#include "gpu_render.cuh"

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            cudaGetErrorString(error)
        );
    }
}

void render_scene(
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
        "Launching path-tracing kernel"
    );

    check_cuda(
        cudaDeviceSynchronize(),
        "Executing path-tracing kernel"
    );
}

int main() {
    GPUDeviceScene lit_device_scene{};
    GPUDeviceScene baseline_device_scene{};

    GPUColor* device_lit_pixels = nullptr;
    GPUColor* device_baseline_pixels = nullptr;

    try {
        constexpr int image_width = 64;
        constexpr int image_height = 64;
        constexpr int pixel_count = image_width * image_height;

        auto diffuse = make_shared<Lambertian>(
            Color(0.8, 0.8, 0.8)
        );

        auto light = make_shared<DiffuseLight>(
            Color(5.0, 5.0, 5.0)
        );

        // Scene containing the diffuse surface and emissive light.
        HittableList lit_world;

        lit_world.add(make_shared<Quad>(
            Point3(-3, -3, 0),
            Vec3(6, 0, 0),
            Vec3(0, 6, 0),
            diffuse
        ));

        lit_world.add(make_shared<Quad>(
            Point3(0.8, 1.8, 2),
            Vec3(1.2, 0, 0),
            Vec3(0, -1.2, 0),
            light
        ));

        // Baseline scene containing only the diffuse surface.
        HittableList baseline_world;

        baseline_world.add(make_shared<Quad>(
            Point3(-3, -3, 0),
            Vec3(6, 0, 0),
            Vec3(0, 6, 0),
            diffuse
        ));

        GPUScene lit_scene =
            build_gpu_scene_from_list(lit_world);

        GPUScene baseline_scene =
            build_gpu_scene_from_list(baseline_world);

        lit_device_scene = upload_gpu_scene(lit_scene);
        baseline_device_scene = upload_gpu_scene(baseline_scene);

        if (lit_device_scene.light_count != 1) {
            throw std::runtime_error(
                "Expected exactly one light in the lit scene."
            );
        }

        if (baseline_device_scene.light_count != 0) {
            throw std::runtime_error(
                "Expected zero lights in the baseline scene."
            );
        }

        Camera camera;

        camera.config.image_width = image_width;
        camera.config.aspect_ratio = 1.0;
        camera.config.samples_per_pixel = 16;
        camera.config.max_ray_bounces = 5;

        camera.config.lookfrom = Point3(0, 0, 5);
        camera.config.lookat = Point3(0, 0, 0);
        camera.config.vup = Vec3(0, 1, 0);

        camera.config.vfov = 40;
        camera.config.defocus_angle = 0;
        camera.config.focus_dist = 5;

        camera.config.background = Color(0.1, 0.2, 0.3);

        camera.initialize();

        GPUCamera gpu_camera = build_gpu_camera(camera);

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_lit_pixels),
                pixel_count * sizeof(GPUColor)
            ),
            "Allocating lit-scene pixels"
        );

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_baseline_pixels),
                pixel_count * sizeof(GPUColor)
            ),
            "Allocating baseline-scene pixels"
        );

        std::vector<GPUColor> lit_pixels(pixel_count);
        std::vector<GPUColor> baseline_pixels(pixel_count);

        // Render both scenes using the same camera and sample seeds.
        render_scene(
            gpu_camera,
            lit_device_scene,
            device_lit_pixels,
            pixel_count
        );

        render_scene(
            gpu_camera,
            baseline_device_scene,
            device_baseline_pixels,
            pixel_count
        );

        check_cuda(
            cudaMemcpy(
                lit_pixels.data(),
                device_lit_pixels,
                pixel_count * sizeof(GPUColor),
                cudaMemcpyDeviceToHost
            ),
            "Copying lit-scene pixels"
        );

        check_cuda(
            cudaMemcpy(
                baseline_pixels.data(),
                device_baseline_pixels,
                pixel_count * sizeof(GPUColor),
                cudaMemcpyDeviceToHost
            ),
            "Copying baseline-scene pixels"
        );

        const int center_index =
            (image_height / 2) * image_width +
            image_width / 2;

        const GPUColor lit_center = lit_pixels[center_index];
        const GPUColor baseline_center =
            baseline_pixels[center_index];

        std::cout << "GPU Lambertian light-contribution test:\n";

        std::cout
            << "Lit center pixel: "
            << lit_center.r << ", "
            << lit_center.g << ", "
            << lit_center.b << '\n';

        std::cout
            << "Baseline center pixel: "
            << baseline_center.r << ", "
            << baseline_center.g << ", "
            << baseline_center.b << '\n';

        const float center_difference =
            std::fabs(lit_center.r - baseline_center.r) +
            std::fabs(lit_center.g - baseline_center.g) +
            std::fabs(lit_center.b - baseline_center.b);

        std::cout
            << "Center pixel difference: "
            << center_difference << '\n';

        const bool center_receives_light =
            center_difference > 1e-5f;

        std::cout
            << "Center receives light contribution: "
            << (center_receives_light ? "PASS" : "FAIL")
            << '\n';

        int changed_pixels = 0;
        int invalid_pixels = 0;

        double total_lit_luminance = 0.0;
        double total_baseline_luminance = 0.0;

        for (int i = 0; i < pixel_count; i++) {
            const GPUColor& lit = lit_pixels[i];
            const GPUColor& baseline = baseline_pixels[i];

            if (
                !std::isfinite(lit.r) ||
                !std::isfinite(lit.g) ||
                !std::isfinite(lit.b) ||
                !std::isfinite(baseline.r) ||
                !std::isfinite(baseline.g) ||
                !std::isfinite(baseline.b)
            ) {
                invalid_pixels++;
                continue;
            }

            const float difference =
                std::fabs(lit.r - baseline.r) +
                std::fabs(lit.g - baseline.g) +
                std::fabs(lit.b - baseline.b);

            if (difference > 1e-5f) {
                changed_pixels++;
            }

            total_lit_luminance +=
                0.2126 * lit.r +
                0.7152 * lit.g +
                0.0722 * lit.b;

            total_baseline_luminance +=
                0.2126 * baseline.r +
                0.7152 * baseline.g +
                0.0722 * baseline.b;
        }

        std::cout
            << "Pixels changed by light: "
            << changed_pixels << " / "
            << pixel_count << '\n';

        std::cout
            << "Average lit luminance: "
            << total_lit_luminance / pixel_count << '\n';

        std::cout
            << "Average baseline luminance: "
            << total_baseline_luminance / pixel_count << '\n';

        std::cout
            << "Invalid pixels: "
            << invalid_pixels << '\n';

        const bool light_contribution =
            changed_pixels > 0 &&
            total_lit_luminance > total_baseline_luminance;

        std::cout
            << "Light contributes to rendering: "
            << (light_contribution ? "PASS" : "FAIL")
            << '\n';

        const bool finite_pixels = invalid_pixels == 0;

        std::cout
            << "Finite pixel values: "
            << (finite_pixels ? "PASS" : "FAIL")
            << '\n';

        check_cuda(
            cudaFree(device_lit_pixels),
            "Freeing lit-scene pixels"
        );
        device_lit_pixels = nullptr;

        check_cuda(
            cudaFree(device_baseline_pixels),
            "Freeing baseline-scene pixels"
        );
        device_baseline_pixels = nullptr;

        free_gpu_scene(lit_device_scene);
        free_gpu_scene(baseline_device_scene);

        std::cout << "GPU Lambertian test completed.\n";

        return (
            center_receives_light &&
            light_contribution &&
            finite_pixels
        ) ? 0 : 1;
    }
    catch (const std::exception& error) {
        if (device_lit_pixels != nullptr) {
            cudaFree(device_lit_pixels);
        }

        if (device_baseline_pixels != nullptr) {
            cudaFree(device_baseline_pixels);
        }

        free_gpu_scene(lit_device_scene);
        free_gpu_scene(baseline_device_scene);

        std::cerr << "ERROR: " << error.what() << '\n';

        return 1;
    }
}