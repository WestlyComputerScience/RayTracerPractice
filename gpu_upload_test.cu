#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>

#include "common_constants.h"
#include "vec3.h"
#include "interval.h"
#include "ray.h"
#include "aabb.h"

#include "gpu_scene_builder.h"
#include "gpu_scene.cuh"
#include "gpu_intersection.cuh"
#include "gpu_camera_builder.h"
#include "gpu_camera.cuh"
#include "gpu_render.cuh"
#include "gpu_traversal.cuh"


bool write_gpu_ppm(
    const std::string& filename,
    const std::vector<GPUColor>& pixels,
    int width,
    int height
) {
    if (width <= 0 || height <= 0 ||
        pixels.size() != static_cast<size_t>(width * height)) {
        return false;
    }

    std::ofstream output(filename);

    if (!output) {
        return false;
    }

    output << "P3\n";
    output << width << " " << height << "\n";
    output << "255\n";

    for (const GPUColor& pixel : pixels) {
        float r = std::isfinite(pixel.r) ? pixel.r : 0.0f;
        float g = std::isfinite(pixel.g) ? pixel.g : 0.0f;
        float b = std::isfinite(pixel.b) ? pixel.b : 0.0f;

        r = std::sqrt(std::max(0.0f, r));
        g = std::sqrt(std::max(0.0f, g));
        b = std::sqrt(std::max(0.0f, b));

        int ir = static_cast<int>(256.0f * std::clamp(r, 0.0f, 0.999f));
        int ig = static_cast<int>(256.0f * std::clamp(g, 0.0f, 0.999f));
        int ib = static_cast<int>(256.0f * std::clamp(b, 0.0f, 0.999f));

        output << ir << ' ' << ig << ' ' << ib << '\n';
    }

    return output.good();
}

__global__ void gpu_bvh_test_kernel(
    const GPUBvhNode* nodes,
    const GPUPrimitive* primitives,
    int node_count,
    int primitive_count,
    const GPURay* rays,
    GPUTraversalResult* results,
    int ray_count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= ray_count) {
        return;
    }

    results[index] = gpu_bvh_hit(
        nodes,
        primitives,
        node_count,
        primitive_count,
        rays[index],
        0.001f,
        1.0e30f
    );
}

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            cudaGetErrorString(error)
        );
    }
}

bool gpu_vec3_matches(const GPUVec3& gpu, const Vec3& cpu) {
    constexpr float tolerance = 1e-5f;

    return
        std::fabs(gpu.x - static_cast<float>(cpu.x())) < tolerance &&
        std::fabs(gpu.y - static_cast<float>(cpu.y())) < tolerance &&
        std::fabs(gpu.z - static_cast<float>(cpu.z())) < tolerance;
}

bool gpu_ray_matches_cpu(
    const GPURay& gpu_ray,
    const Ray& cpu_ray
) {
    constexpr float tolerance = 1e-3f;

    return
        std::fabs(gpu_ray.origin_x - static_cast<float>(cpu_ray.origin().x())) < tolerance &&
        std::fabs(gpu_ray.origin_y - static_cast<float>(cpu_ray.origin().y())) < tolerance &&
        std::fabs(gpu_ray.origin_z - static_cast<float>(cpu_ray.origin().z())) < tolerance &&

        std::fabs(gpu_ray.direction_x - static_cast<float>(cpu_ray.direction().x())) < tolerance &&
        std::fabs(gpu_ray.direction_y - static_cast<float>(cpu_ray.direction().y())) < tolerance &&
        std::fabs(gpu_ray.direction_z - static_cast<float>(cpu_ray.direction().z())) < tolerance;
}


void test_gpu_scene_render(const GPUDeviceScene& device_scene) {
    // Create a separate camera aimed at the four test quads.
    Camera scene_camera;

    scene_camera.config.image_width = 400;
    scene_camera.config.aspect_ratio = 16.0 / 9.0;
    scene_camera.config.samples_per_pixel = 1;
    scene_camera.config.max_ray_bounces = 1;

    scene_camera.config.lookfrom = Point3(5, 1, -10);
    scene_camera.config.lookat = Point3(5, 1, 0);
    scene_camera.config.vup = Vec3(0, 1, 0);

    scene_camera.config.vfov = 40;
    scene_camera.config.defocus_angle = 0;
    scene_camera.config.focus_dist = 10;

    scene_camera.config.background = Color(0.1, 0.2, 0.3);

    scene_camera.initialize();

    GPUCamera scene_gpu_camera = build_gpu_camera(scene_camera);

    int pixel_count =
        scene_gpu_camera.image_width * scene_gpu_camera.image_height;

    GPUColor* device_pixels = nullptr;

    check_cuda(
        cudaMalloc(
            reinterpret_cast<void**>(&device_pixels),
            pixel_count * sizeof(GPUColor)
        ),
        "Allocating scene-render pixels"
    );

    constexpr int threads_per_block = 256;

    int blocks =
        (pixel_count + threads_per_block - 1) / threads_per_block;

    std::vector<GPUColor> pixels(pixel_count);

    constexpr float tolerance = 1e-5f;

    // ------------------------------------------------------------
    // Test 1: Render the uploaded scene using flat material colors.
    // ------------------------------------------------------------

    gpu_render_scene<<<blocks, threads_per_block>>>(
        scene_gpu_camera,
        device_scene.bvh_nodes,
        device_scene.primitives,
        device_scene.materials,
        static_cast<int>(device_scene.bvh_node_count),
        static_cast<int>(device_scene.primitive_count),
        static_cast<int>(device_scene.material_count),
        device_pixels
    );

    check_cuda(cudaGetLastError(), "Launching GPU scene-render kernel");
    check_cuda(cudaDeviceSynchronize(), "Executing GPU scene-render kernel");

    check_cuda(
        cudaMemcpy(
            pixels.data(),
            device_pixels,
            pixel_count * sizeof(GPUColor),
            cudaMemcpyDeviceToHost
        ),
        "Copying rendered scene pixels"
    );

    int red_pixels = 0;
    int green_pixels = 0;
    int light_pixels = 0;
    int background_pixels = 0;
    int diagnostic_pixels = 0;
    int unknown_pixels = 0;

    for (const GPUColor& pixel : pixels) {
        bool is_red =
            std::fabs(pixel.r - 1.0f) < tolerance &&
            std::fabs(pixel.g) < tolerance &&
            std::fabs(pixel.b) < tolerance;

        bool is_green =
            std::fabs(pixel.r) < tolerance &&
            std::fabs(pixel.g - 1.0f) < tolerance &&
            std::fabs(pixel.b) < tolerance;

        bool is_background =
            std::fabs(pixel.r - scene_gpu_camera.background_r) < tolerance &&
            std::fabs(pixel.g - scene_gpu_camera.background_g) < tolerance &&
            std::fabs(pixel.b - scene_gpu_camera.background_b) < tolerance;

        bool is_diagnostic =
            std::fabs(pixel.r - 1.0f) < tolerance &&
            std::fabs(pixel.g) < tolerance &&
            std::fabs(pixel.b - 1.0f) < tolerance;
        
        bool is_light =
            std::fabs(pixel.r - 5.0f) < tolerance &&
            std::fabs(pixel.g - 5.0f) < tolerance &&
            std::fabs(pixel.b - 5.0f) < tolerance;

        if (is_red) {
            red_pixels++;
        }
        else if (is_green) {
            green_pixels++;
        }
        else if (is_light) {
            light_pixels++;
        }
        else if (is_background) {
            background_pixels++;
        }
        else if (is_diagnostic) {
            diagnostic_pixels++;
        }
        else {
            unknown_pixels++;
        }
    }

    std::cout << "\nGPU scene-rendering results:\n";
    std::cout << "Red pixels: " << red_pixels << '\n';
    std::cout << "Green pixels: " << green_pixels << '\n';
    std::cout << "Light pixels: " << light_pixels << '\n';
    std::cout << "Background pixels: " << background_pixels << '\n';
    std::cout << "Diagnostic pixels: " << diagnostic_pixels << '\n';
    std::cout << "Unknown pixels: " << unknown_pixels << '\n';

    bool scene_passed =
        red_pixels > 0 &&
        green_pixels > 0 &&
        background_pixels > 0 &&
        diagnostic_pixels == 0 &&
        unknown_pixels == 0;

    std::cout << "GPU scene rendering: "
              << (scene_passed ? "PASS" : "FAIL")
              << '\n';

    if (!scene_passed) {
        cudaFree(device_pixels);
        throw std::runtime_error("GPU scene-rendering test failed.");
    }

    
    // ------------------------------------------------------------
    // Test 2: Render the scene using surface normal visualization.
    // ------------------------------------------------------------

    gpu_render_normals<<<blocks, threads_per_block>>>(
        scene_gpu_camera,
        device_scene.bvh_nodes,
        device_scene.primitives,
        static_cast<int>(device_scene.bvh_node_count),
        static_cast<int>(device_scene.primitive_count),
        device_pixels
    );

    check_cuda(cudaGetLastError(), "Launching GPU normal-rendering kernel");
    check_cuda(cudaDeviceSynchronize(), "Executing GPU normal-rendering kernel");

    check_cuda(
        cudaMemcpy(
            pixels.data(),
            device_pixels,
            pixel_count * sizeof(GPUColor),
            cudaMemcpyDeviceToHost
        ),
        "Copying normal-rendered pixels"
    );

    int normal_pixels = 0;
    int light_normal_pixels = 0;
    background_pixels = 0;
    diagnostic_pixels = 0;
    unknown_pixels = 0;

    for (const GPUColor& pixel : pixels) {
        bool is_normal =
            std::fabs(pixel.r - 0.5f) < tolerance &&
            std::fabs(pixel.g - 0.5f) < tolerance &&
            std::fabs(pixel.b - 1.0f) < tolerance;

        // The light quad has a geometric normal of (0, -1, 0).
        // Its normal visualization should therefore be (0.5, 0, 0.5).
        bool is_light_normal =
            std::fabs(pixel.r - 0.5f) < tolerance &&
            std::fabs(pixel.g) < tolerance &&
            std::fabs(pixel.b - 0.5f) < tolerance;

        bool is_background =
            std::fabs(pixel.r - scene_gpu_camera.background_r) < tolerance &&
            std::fabs(pixel.g - scene_gpu_camera.background_g) < tolerance &&
            std::fabs(pixel.b - scene_gpu_camera.background_b) < tolerance;

        bool is_diagnostic =
            std::fabs(pixel.r - 1.0f) < tolerance &&
            std::fabs(pixel.g) < tolerance &&
            std::fabs(pixel.b - 1.0f) < tolerance;

        if (is_normal) {
            normal_pixels++;
        }
        else if (is_light_normal) {
            light_normal_pixels++;
        }
        else if (is_background) {
            background_pixels++;
        }
        else if (is_diagnostic) {
            diagnostic_pixels++;
        }
        else {
            unknown_pixels++;
        }
    }

    std::cout << "\nGPU normal-rendering results:\n";
    std::cout << "Normal pixels: " << normal_pixels << '\n';
    std::cout << "Light normal pixels: " << light_normal_pixels << '\n';
    std::cout << "Background pixels: " << background_pixels << '\n';
    std::cout << "Diagnostic pixels: " << diagnostic_pixels << '\n';
    std::cout << "Unknown pixels: " << unknown_pixels << '\n';

    bool normals_passed =
        normal_pixels > 0 &&
        light_normal_pixels > 0 &&
        background_pixels > 0 &&
        diagnostic_pixels == 0 &&
        unknown_pixels == 0;

    std::cout << "GPU normal rendering: "
              << (normals_passed ? "PASS" : "FAIL")
              << '\n';

    if (!normals_passed) {
        cudaFree(device_pixels);
        throw std::runtime_error("GPU normal-rendering test failed.");
    }


        // ------------------------------------------------------------
    // Test 3: Render the scene using Lambertian diffuse lighting.
    // ------------------------------------------------------------

    gpu_render_diffuse<<<blocks, threads_per_block>>>(
        scene_gpu_camera,
        device_scene.bvh_nodes,
        device_scene.primitives,
        device_scene.materials,
        static_cast<int>(device_scene.bvh_node_count),
        static_cast<int>(device_scene.primitive_count),
        static_cast<int>(device_scene.material_count),
        device_pixels
    );

    check_cuda(
        cudaGetLastError(),
        "Launching GPU diffuse-rendering kernel"
    );

    check_cuda(
        cudaDeviceSynchronize(),
        "Executing GPU diffuse-rendering kernel"
    );

    check_cuda(
        cudaMemcpy(
            pixels.data(),
            device_pixels,
            pixel_count * sizeof(GPUColor),
            cudaMemcpyDeviceToHost
        ),
        "Copying diffuse-rendered pixels"
    );

    constexpr float expected_diffuse = 0.81649658f;

    int diffuse_red_pixels = 0;
    int diffuse_green_pixels = 0;

    // Reuse light_pixels, which was declared in Test 1.
    light_pixels = 0;
    background_pixels = 0;
    diagnostic_pixels = 0;
    unknown_pixels = 0;

    for (const GPUColor& pixel : pixels) {
        bool is_diffuse_red =
            std::fabs(pixel.r - expected_diffuse) < tolerance &&
            std::fabs(pixel.g) < tolerance &&
            std::fabs(pixel.b) < tolerance;

        bool is_diffuse_green =
            std::fabs(pixel.r) < tolerance &&
            std::fabs(pixel.g - expected_diffuse) < tolerance &&
            std::fabs(pixel.b) < tolerance;

        bool is_light =
            std::fabs(pixel.r - 5.0f) < tolerance &&
            std::fabs(pixel.g - 5.0f) < tolerance &&
            std::fabs(pixel.b - 5.0f) < tolerance;

        bool is_background =
            std::fabs(pixel.r - scene_gpu_camera.background_r) < tolerance &&
            std::fabs(pixel.g - scene_gpu_camera.background_g) < tolerance &&
            std::fabs(pixel.b - scene_gpu_camera.background_b) < tolerance;

        bool is_diagnostic =
            std::fabs(pixel.r - 1.0f) < tolerance &&
            std::fabs(pixel.g) < tolerance &&
            std::fabs(pixel.b - 1.0f) < tolerance;

        if (is_diffuse_red) {
            diffuse_red_pixels++;
        }
        else if (is_diffuse_green) {
            diffuse_green_pixels++;
        }
        else if (is_light) {
            light_pixels++;
        }
        else if (is_background) {
            background_pixels++;
        }
        else if (is_diagnostic) {
            diagnostic_pixels++;
        }
        else {
            unknown_pixels++;
        }
    }

    std::cout << "\nGPU diffuse-rendering results:\n";
    std::cout << "Diffuse red pixels: " << diffuse_red_pixels << '\n';
    std::cout << "Diffuse green pixels: " << diffuse_green_pixels << '\n';
    std::cout << "Light pixels: " << light_pixels << '\n';
    std::cout << "Background pixels: " << background_pixels << '\n';
    std::cout << "Diagnostic pixels: " << diagnostic_pixels << '\n';
    std::cout << "Unknown pixels: " << unknown_pixels << '\n';

    std::cout << "Expected diffuse intensity: "
              << expected_diffuse << '\n';

    bool diffuse_passed =
        diffuse_red_pixels > 0 &&
        diffuse_green_pixels > 0 &&
        light_pixels > 0 &&
        background_pixels > 0 &&
        diagnostic_pixels == 0 &&
        unknown_pixels == 0;

    std::cout << "GPU diffuse rendering: "
              << (diffuse_passed ? "PASS" : "FAIL")
              << '\n';

    cudaFree(device_pixels);

    if (!diffuse_passed) {
        throw std::runtime_error("GPU diffuse-rendering test failed.");
    }
}


void test_gpu_depth_of_field() {
    // Create three colored targets at different distances.
    HittableList world;

    auto red_material = make_shared<Lambertian>(
        Color(0.9, 0.08, 0.08)
    );

    auto green_material = make_shared<Lambertian>(
        Color(0.08, 0.9, 0.08)
    );

    auto blue_material = make_shared<Lambertian>(
        Color(0.08, 0.15, 0.9)
    );

    // Focus target: 8 units from the camera.
    world.add(make_shared<Quad>(
        Point3(-0.9, -1.3, 0),
        Vec3(1.8, 0, 0),
        Vec3(0, 2.6, 0),
        red_material
    ));

    // Background target: 11 units from the camera.
    world.add(make_shared<Quad>(
        Point3(-3.08 - 0.99, -1.54, 3),
        Vec3(1.98, 0, 0),
        Vec3(0, 3.08, 0),
        green_material
    ));

    // Foreground target: 14 units from the camera.
    world.add(make_shared<Quad>(
        Point3(3.92 - 1.26, -1.96, 6),
        Vec3(2.52, 0, 0),
        Vec3(0, 3.92, 0),
        blue_material
    ));

    // Build and upload the GPU scene.
    GPUScene scene = build_gpu_scene_from_list(world);
    GPUDeviceScene device_scene = upload_gpu_scene(scene);

    std::cout << "\nGPU depth-of-field scene:\n";
    std::cout << "BVH nodes: " << device_scene.bvh_node_count << '\n';
    std::cout << "Primitives: " << device_scene.primitive_count << '\n';

    // Configure the camera.
    Camera camera;

    camera.config.image_width = 600;
    camera.config.aspect_ratio = 16.0 / 9.0;
    camera.config.samples_per_pixel = 64;
    camera.config.max_ray_bounces = 1;

    camera.config.lookfrom = Point3(0, 0, -8);
    camera.config.lookat = Point3(0, 0, 0);
    camera.config.vup = Vec3(0, 1, 0);

    camera.config.vfov = 40;
    camera.config.focus_dist = 8;

    camera.config.background = Color(0.025, 0.025, 0.025);

    const int image_width = camera.config.image_width;
    const int image_height = static_cast<int>(
        image_width / camera.config.aspect_ratio
    );

    const int pixel_count = image_width * image_height;

    GPUColor* device_pixels = nullptr;

    check_cuda(
        cudaMalloc(
            &device_pixels,
            pixel_count * sizeof(GPUColor)
        ),
        "Allocating depth-of-field render pixels"
    );

    const int threads_per_block = 256;
    const int blocks =
        (pixel_count + threads_per_block - 1) / threads_per_block;

    // Render once with the specified aperture setting.
    auto render = [&](real defocus_angle, const std::string& filename) {
        camera.config.defocus_angle = defocus_angle;
        camera.initialize();

        GPUCamera gpu_camera = build_gpu_camera(camera);

        gpu_render_scene_multisample<<<blocks, threads_per_block>>>(
            gpu_camera,
            device_scene.bvh_nodes,
            device_scene.primitives,
            device_scene.materials,
            static_cast<int>(device_scene.bvh_node_count),
            static_cast<int>(device_scene.primitive_count),
            static_cast<int>(device_scene.material_count),
            device_pixels
        );

        check_cuda(
            cudaGetLastError(),
            "Launching multisample depth-of-field renderer"
        );

        check_cuda(
            cudaDeviceSynchronize(),
            "Synchronizing depth-of-field renderer"
        );

        std::vector<GPUColor> pixels(pixel_count);

        check_cuda(
            cudaMemcpy(
                pixels.data(),
                device_pixels,
                pixel_count * sizeof(GPUColor),
                cudaMemcpyDeviceToHost
            ),
            "Copying depth-of-field render to CPU"
        );

        if (!write_gpu_ppm(
            filename,
            pixels,
            image_width,
            image_height
        )) {
            throw std::runtime_error(
                "Failed to write image: " + filename
            );
        }

        std::cout
            << "Wrote " << filename
            << " (defocus angle: " << defocus_angle << ")\n";
    };

    try {
        render(0, "gpu_defocus_off.ppm");
        render(15, "gpu_defocus_on.ppm");
    } catch (...) {
        cudaFree(device_pixels);
        free_gpu_scene(device_scene);
        throw;
    }

    check_cuda(
        cudaFree(device_pixels),
        "Freeing depth-of-field render pixels"
    );

    free_gpu_scene(device_scene);

    std::cout << "GPU depth-of-field test: PASS\n";
}

int main() {
    try {
        // Initialize a CPU camera with known settings.
        Camera camera;

        camera.config.image_width = 800;
        camera.config.aspect_ratio = 16.0 / 9.0;
        camera.config.samples_per_pixel = 100;
        camera.config.max_ray_bounces = 20;

        camera.config.lookfrom = Point3(278, 278, -800);
        camera.config.lookat = Point3(278, 278, 0);
        camera.config.vup = Vec3(0, 1, 0);

        camera.config.vfov = 40;
        camera.config.defocus_angle = 0;
        camera.config.focus_dist = 10;

        camera.config.background = Color(0.1, 0.2, 0.3);

        // Calculate the CPU camera's geometry.
        camera.initialize();

        // Convert the initialized camera into GPU-compatible data.
        GPUCamera gpu_camera = build_gpu_camera(camera);

        // Test GPU camera ray generation.
        int pixel_count = gpu_camera.image_width * gpu_camera.image_height;

        GPURay* device_rays = nullptr;

        cudaError_t cuda_status = cudaMalloc(
            &device_rays,
            pixel_count * sizeof(GPURay)
        );

        if (cuda_status != cudaSuccess) {
            throw std::runtime_error(
                "Failed to allocate GPU camera rays: " +
                std::string(cudaGetErrorString(cuda_status))
            );
        }

        constexpr int threads_per_block = 256;

        int blocks =
            (pixel_count + threads_per_block - 1) / threads_per_block;

        gpu_generate_camera_rays<<<blocks, threads_per_block>>>(
            gpu_camera,
            device_rays
        );

        cuda_status = cudaGetLastError();

        if (cuda_status != cudaSuccess) {
            cudaFree(device_rays);

            throw std::runtime_error(
                "GPU camera ray kernel launch failed: " +
                std::string(cudaGetErrorString(cuda_status))
            );
        }

        cuda_status = cudaDeviceSynchronize();

        if (cuda_status != cudaSuccess) {
            cudaFree(device_rays);

            throw std::runtime_error(
                "GPU camera ray kernel execution failed: " +
                std::string(cudaGetErrorString(cuda_status))
            );
        }

        std::vector<GPURay> gpu_rays(pixel_count);

        cuda_status = cudaMemcpy(
            gpu_rays.data(),
            device_rays,
            pixel_count * sizeof(GPURay),
            cudaMemcpyDeviceToHost
        );

        cudaFree(device_rays);

        if (cuda_status != cudaSuccess) {
            throw std::runtime_error(
                "Failed to copy GPU camera rays: " +
                std::string(cudaGetErrorString(cuda_status))
            );
        }

        // Compare three representative pixels.
        int test_pixels[] = {
            0,
            pixel_count / 2,
            pixel_count - 1
        };

        bool rays_match = true;

        for (int pixel_index : test_pixels) {
            int i = pixel_index % camera.config.image_width;
            int j = pixel_index / camera.config.image_width;

            Point3 pixel_center =
                camera.get_pixel00_loc() +
                i * camera.get_pixel_delta_u() +
                j * camera.get_pixel_delta_v();

            Ray cpu_ray(
                camera.get_center(),
                pixel_center - camera.get_center()
            );

            const GPURay& gpu = gpu_rays[pixel_index];

            bool matches = gpu_ray_matches_cpu(gpu, cpu_ray);

            std::cout
                << "Pixel " << pixel_index
                << " (" << i << ", " << j << "): "
                << (matches ? "MATCH" : "MISMATCH")
                << '\n';

            if (!matches) {
                rays_match = false;

                std::cout << "CPU origin: "
                        << cpu_ray.origin().x() << ", "
                        << cpu_ray.origin().y() << ", "
                        << cpu_ray.origin().z() << '\n';

                std::cout << "GPU origin: "
                        << gpu.origin_x << ", "
                        << gpu.origin_y << ", "
                        << gpu.origin_z << '\n';

                std::cout << "CPU direction: "
                        << cpu_ray.direction().x() << ", "
                        << cpu_ray.direction().y() << ", "
                        << cpu_ray.direction().z() << '\n';

                std::cout << "GPU direction: "
                        << gpu.direction_x << ", "
                        << gpu.direction_y << ", "
                        << gpu.direction_z << '\n';
            }
        }

        std::cout
            << "CPU vs GPU camera ray generation: "
            << (rays_match ? "MATCH" : "MISMATCH")
            << '\n';

        // Test stratified multi-sample GPU camera ray generation.
        GPUCamera sample_camera = gpu_camera;

        sample_camera.image_width = 4;
        sample_camera.image_height = 3;
        sample_camera.samples_per_pixel = 4;
        sample_camera.sqrt_spp = 2;
        sample_camera.recip_sqrt_spp = 0.5f;

        int sample_pixel_count =
            sample_camera.image_width * sample_camera.image_height;

        int sample_count_per_pixel =
            sample_camera.sqrt_spp * sample_camera.sqrt_spp;

        int total_sample_count =
            sample_pixel_count * sample_count_per_pixel;

        GPURay* device_sample_rays = nullptr;

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_sample_rays),
                total_sample_count * sizeof(GPURay)
            ),
            "cudaMalloc multi-sample camera rays"
        );

        int sample_blocks =
            (total_sample_count + threads_per_block - 1)
            / threads_per_block;

        gpu_generate_camera_sample_rays<<<
            sample_blocks,
            threads_per_block
        >>>(
            sample_camera,
            device_sample_rays
        );

        check_cuda(
            cudaGetLastError(),
            "multi-sample camera kernel launch"
        );

        check_cuda(
            cudaDeviceSynchronize(),
            "multi-sample camera synchronization"
        );

        std::vector<GPURay> sample_rays(total_sample_count);

        check_cuda(
            cudaMemcpy(
                sample_rays.data(),
                device_sample_rays,
                total_sample_count * sizeof(GPURay),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy multi-sample camera rays"
        );

        cudaFree(device_sample_rays);

        std::cout
            << "GPU multi-sample camera ray generation: "
            << sample_rays.size() << " rays generated"
            << '\n';

                    // Validate that every sample lies inside its expected
        // stratified subpixel cell.
        bool stratified_samples_match = true;

        float delta_u_length_squared =
            sample_camera.pixel_delta_u.x * sample_camera.pixel_delta_u.x +
            sample_camera.pixel_delta_u.y * sample_camera.pixel_delta_u.y +
            sample_camera.pixel_delta_u.z * sample_camera.pixel_delta_u.z;

        float delta_v_length_squared =
            sample_camera.pixel_delta_v.x * sample_camera.pixel_delta_v.x +
            sample_camera.pixel_delta_v.y * sample_camera.pixel_delta_v.y +
            sample_camera.pixel_delta_v.z * sample_camera.pixel_delta_v.z;

        for (int index = 0; index < total_sample_count; index++) {
            int pixel_index = index / sample_count_per_pixel;

            int i = pixel_index % sample_camera.image_width;
            int j = pixel_index / sample_camera.image_width;

            int sample_index = index % sample_count_per_pixel;

            int s_i = sample_index % sample_camera.sqrt_spp;
            int s_j = sample_index / sample_camera.sqrt_spp;

            const GPURay& ray = sample_rays[index];

            // Recover the sampled pixel position from the ray.
            float pixel_x = ray.origin_x + ray.direction_x;
            float pixel_y = ray.origin_y + ray.direction_y;
            float pixel_z = ray.origin_z + ray.direction_z;

            float relative_x =
                pixel_x - sample_camera.pixel00_loc.x;

            float relative_y =
                pixel_y - sample_camera.pixel00_loc.y;

            float relative_z =
                pixel_z - sample_camera.pixel00_loc.z;

            // Project the sampled position onto the camera's
            // horizontal and vertical pixel-delta vectors.
            float pixel_coordinate_x =
                (
                    relative_x * sample_camera.pixel_delta_u.x +
                    relative_y * sample_camera.pixel_delta_u.y +
                    relative_z * sample_camera.pixel_delta_u.z
                ) / delta_u_length_squared;

            float pixel_coordinate_y =
                (
                    relative_x * sample_camera.pixel_delta_v.x +
                    relative_y * sample_camera.pixel_delta_v.y +
                    relative_z * sample_camera.pixel_delta_v.z
                ) / delta_v_length_squared;

            float offset_x = pixel_coordinate_x - i;
            float offset_y = pixel_coordinate_y - j;

            float lower_x =
                -0.5f + s_i * sample_camera.recip_sqrt_spp;

            float upper_x =
                -0.5f + (s_i + 1) * sample_camera.recip_sqrt_spp;

            float lower_y =
                -0.5f + s_j * sample_camera.recip_sqrt_spp;

            float upper_y =
                -0.5f + (s_j + 1) * sample_camera.recip_sqrt_spp;

            constexpr float tolerance = 1e-3f;

            bool sample_is_valid =
                offset_x >= lower_x - tolerance &&
                offset_x < upper_x + tolerance &&
                offset_y >= lower_y - tolerance &&
                offset_y < upper_y + tolerance;

            if (!sample_is_valid) {
                std::cout
                    << "Stratified sample mismatch at index "
                    << index
                    << ": pixel (" << i << ", " << j << ")"
                    << ", stratum (" << s_i << ", " << s_j << ")"
                    << ", offset (" << offset_x << ", " << offset_y << ")"
                    << '\n';

                stratified_samples_match = false;
                break;
            }
        }

        std::cout
            << "GPU stratified sampling: "
            << (stratified_samples_match ? "PASS" : "FAIL")
            << '\n';

                // Test GPU defocus-disk sampling.
        Camera defocus_test_camera = camera;
        defocus_test_camera.config.defocus_angle = 10.0;

        defocus_test_camera.initialize();

        GPUCamera defocus_gpu_camera =
            build_gpu_camera(defocus_test_camera);

        // Limit this test to exactly 1,000 generated rays.
        defocus_gpu_camera.image_width = 10;
        defocus_gpu_camera.image_height = 1;
        defocus_gpu_camera.samples_per_pixel = 100;
        defocus_gpu_camera.sqrt_spp = 10;
        defocus_gpu_camera.recip_sqrt_spp = 0.1f;

        std::cout
            << "Defocus angle: " << defocus_gpu_camera.defocus_angle
            << '\n';

        std::cout
            << "Defocus disk U: "
            << defocus_gpu_camera.defocus_disk_u.x << ", "
            << defocus_gpu_camera.defocus_disk_u.y << ", "
            << defocus_gpu_camera.defocus_disk_u.z
            << '\n';

        std::cout
            << "Defocus disk V: "
            << defocus_gpu_camera.defocus_disk_v.x << ", "
            << defocus_gpu_camera.defocus_disk_v.y << ", "
            << defocus_gpu_camera.defocus_disk_v.z
            << '\n';

                constexpr int defocus_sample_count = 1000;

        GPURay* device_defocus_rays = nullptr;

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_defocus_rays),
                defocus_sample_count * sizeof(GPURay)
            ),
            "cudaMalloc defocus sample rays"
        );

        gpu_generate_camera_sample_rays<<<
            (defocus_sample_count + 255) / 256,
            256
        >>>(
            defocus_gpu_camera,
            device_defocus_rays
        );

        check_cuda(
            cudaGetLastError(),
            "defocus sample kernel launch"
        );

        check_cuda(
            cudaDeviceSynchronize(),
            "defocus sample synchronization"
        );

        std::vector<GPURay> defocus_rays(defocus_sample_count);

        check_cuda(
            cudaMemcpy(
                defocus_rays.data(),
                device_defocus_rays,
                defocus_sample_count * sizeof(GPURay),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy defocus sample rays"
        );

        cudaFree(device_defocus_rays);

        int defocus_noncenter_count = 0;
int defocus_invalid_count = 0;
int direction_invalid_count = 0;

float maximum_radius_squared = 0.0f;

float disk_u_length_squared =
    defocus_gpu_camera.defocus_disk_u.x *
        defocus_gpu_camera.defocus_disk_u.x +
    defocus_gpu_camera.defocus_disk_u.y *
        defocus_gpu_camera.defocus_disk_u.y +
    defocus_gpu_camera.defocus_disk_u.z *
        defocus_gpu_camera.defocus_disk_u.z;

float disk_v_length_squared =
    defocus_gpu_camera.defocus_disk_v.x *
        defocus_gpu_camera.defocus_disk_v.x +
    defocus_gpu_camera.defocus_disk_v.y *
        defocus_gpu_camera.defocus_disk_v.y +
    defocus_gpu_camera.defocus_disk_v.z *
        defocus_gpu_camera.defocus_disk_v.z;

int defocus_sample_count_per_pixel =
    defocus_gpu_camera.sqrt_spp *
    defocus_gpu_camera.sqrt_spp;

for (int index = 0; index < defocus_sample_count; index++) {
    const GPURay& ray = defocus_rays[index];

    // --------------------------------------------------------
    // 1. Validate the ray origin against the defocus disk.
    // --------------------------------------------------------

    float offset_x =
        ray.origin_x - defocus_gpu_camera.center.x;

    float offset_y =
        ray.origin_y - defocus_gpu_camera.center.y;

    float offset_z =
        ray.origin_z - defocus_gpu_camera.center.z;

    float disk_x =
        (offset_x * defocus_gpu_camera.defocus_disk_u.x +
         offset_y * defocus_gpu_camera.defocus_disk_u.y +
         offset_z * defocus_gpu_camera.defocus_disk_u.z) /
        disk_u_length_squared;

    float disk_y =
        (offset_x * defocus_gpu_camera.defocus_disk_v.x +
         offset_y * defocus_gpu_camera.defocus_disk_v.y +
         offset_z * defocus_gpu_camera.defocus_disk_v.z) /
        disk_v_length_squared;

    float radius_squared =
        disk_x * disk_x + disk_y * disk_y;

    if (radius_squared > maximum_radius_squared) {
        maximum_radius_squared = radius_squared;
    }

    if (radius_squared > 1.00001f) {
        defocus_invalid_count++;
    }

    if (offset_x * offset_x +
        offset_y * offset_y +
        offset_z * offset_z > 1e-10f) {
        defocus_noncenter_count++;
    }

    // --------------------------------------------------------
    // 2. Recover the pixel sample targeted by this ray.
    // --------------------------------------------------------

    float pixel_x = ray.origin_x + ray.direction_x;
    float pixel_y = ray.origin_y + ray.direction_y;
    float pixel_z = ray.origin_z + ray.direction_z;

    float relative_x =
        pixel_x - defocus_gpu_camera.pixel00_loc.x;

    float relative_y =
        pixel_y - defocus_gpu_camera.pixel00_loc.y;

    float relative_z =
        pixel_z - defocus_gpu_camera.pixel00_loc.z;

    float pixel_coordinate_x =
        (
            relative_x * defocus_gpu_camera.pixel_delta_u.x +
            relative_y * defocus_gpu_camera.pixel_delta_u.y +
            relative_z * defocus_gpu_camera.pixel_delta_u.z
        ) / delta_u_length_squared;

    float pixel_coordinate_y =
        (
            relative_x * defocus_gpu_camera.pixel_delta_v.x +
            relative_y * defocus_gpu_camera.pixel_delta_v.y +
            relative_z * defocus_gpu_camera.pixel_delta_v.z
        ) / delta_v_length_squared;

    // --------------------------------------------------------
    // 3. Determine which pixel and stratum generated this ray.
    // --------------------------------------------------------

    int pixel_index = index / defocus_sample_count_per_pixel;

    int i = pixel_index % defocus_gpu_camera.image_width;
    int j = pixel_index / defocus_gpu_camera.image_width;

    int sample_index = index % defocus_sample_count_per_pixel;

    int s_i = sample_index % defocus_gpu_camera.sqrt_spp;
    int s_j = sample_index / defocus_gpu_camera.sqrt_spp;

    float sample_offset_x = pixel_coordinate_x - i;
    float sample_offset_y = pixel_coordinate_y - j;

    float lower_x =
        -0.5f + s_i * defocus_gpu_camera.recip_sqrt_spp;

    float upper_x =
        -0.5f + (s_i + 1) * defocus_gpu_camera.recip_sqrt_spp;

    float lower_y =
        -0.5f + s_j * defocus_gpu_camera.recip_sqrt_spp;

    float upper_y =
        -0.5f + (s_j + 1) * defocus_gpu_camera.recip_sqrt_spp;

    constexpr float tolerance = 1e-3f;

    bool direction_is_valid =
        sample_offset_x >= lower_x - tolerance &&
        sample_offset_x < upper_x + tolerance &&
        sample_offset_y >= lower_y - tolerance &&
        sample_offset_y < upper_y + tolerance;

    if (!direction_is_valid) {
        direction_invalid_count++;
    }
        }

        std::cout
            << "GPU defocus samples: "
            << defocus_rays.size()
            << '\n';

        std::cout
            << "Non-center ray origins: "
            << defocus_noncenter_count
            << '\n';

        std::cout
            << "Invalid defocus samples: "
            << defocus_invalid_count
            << '\n';

        std::cout
            << "Maximum disk radius squared: "
            << maximum_radius_squared
            << '\n';

        std::cout
            << "Invalid ray directions: "
            << direction_invalid_count
            << '\n';

        bool defocus_passed =
            defocus_noncenter_count == defocus_sample_count &&
            defocus_invalid_count == 0 &&
            direction_invalid_count == 0;

        std::cout
            << "GPU defocus-disk sampling: "
            << (defocus_passed ? "PASS" : "FAIL")
            << '\n';

        // Continue with the existing background-rendering test.
        GPUColor* device_pixels = nullptr;

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_pixels),
                pixel_count * sizeof(GPUColor)
            ),
            "cudaMalloc background pixels"
        );

        int render_blocks = (pixel_count + 255) / 256;

        gpu_render_background<<<render_blocks, 256>>>(
            gpu_camera,
            device_pixels
        );

        check_cuda(
            cudaGetLastError(),
            "gpu_render_background kernel launch"
        );

        check_cuda(
            cudaDeviceSynchronize(),
            "gpu_render_background synchronization"
        );

        std::vector<GPUColor> gpu_pixels(pixel_count);

        check_cuda(
            cudaMemcpy(
                gpu_pixels.data(),
                device_pixels,
                pixel_count * sizeof(GPUColor),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy background pixels"
        );

        cudaFree(device_pixels);

        // Compare every GPU pixel against the expected background.
        bool background_matches = true;

        for (int i = 0; i < pixel_count; i++) {
            if (
                std::fabs(gpu_pixels[i].r - static_cast<float>(camera.config.background.r())) > 1e-6f ||
                std::fabs(gpu_pixels[i].g - static_cast<float>(camera.config.background.g())) > 1e-6f ||
                std::fabs(gpu_pixels[i].b - static_cast<float>(camera.config.background.b())) > 1e-6f
            ) {
                std::cout << "Background mismatch at pixel " << i
                        << ": GPU = ("
                        << gpu_pixels[i].r << ", "
                        << gpu_pixels[i].g << ", "
                        << gpu_pixels[i].b << ")\n";

                background_matches = false;
                break;
            }
        }

        std::cout << "CPU vs GPU background rendering: "
                << (background_matches ? "MATCH" : "MISMATCH")
                << "\n";

        if (!background_matches) {
            return 1;
        }

        if (!rays_match) {
            throw std::runtime_error(
                "GPU camera ray generation test failed."
            );
        }

        // Compare the GPU geometry against the CPU geometry.
        bool camera_matches =
            gpu_vec3_matches(gpu_camera.center, camera.get_center()) &&
            gpu_vec3_matches(gpu_camera.pixel00_loc, camera.get_pixel00_loc()) &&
            gpu_vec3_matches(gpu_camera.pixel_delta_u, camera.get_pixel_delta_u()) &&
            gpu_vec3_matches(gpu_camera.pixel_delta_v, camera.get_pixel_delta_v()) &&
            gpu_vec3_matches(gpu_camera.defocus_disk_u, camera.get_defocus_disk_u()) &&
            gpu_vec3_matches(gpu_camera.defocus_disk_v, camera.get_defocus_disk_v());

        // Check the remaining camera settings.
        camera_matches =
            camera_matches &&
            gpu_camera.image_width == camera.config.image_width &&
            gpu_camera.image_height == camera.get_image_height() &&
            gpu_camera.samples_per_pixel == camera.config.samples_per_pixel &&
            gpu_camera.max_ray_bounces == camera.config.max_ray_bounces &&
            gpu_camera.defocus_angle == static_cast<float>(camera.config.defocus_angle) &&
            gpu_camera.background_r == static_cast<float>(camera.config.background.r()) &&
            gpu_camera.background_g == static_cast<float>(camera.config.background.g()) &&
            gpu_camera.background_b == static_cast<float>(camera.config.background.b());

        std::cout << "CPU vs GPU camera conversion: "
                << (camera_matches ? "MATCH" : "MISMATCH")
                << '\n';

        if (!camera_matches) {
            throw std::runtime_error("GPU camera conversion test failed.");
        }

        // Create a CPU scene containing four quads.
        HittableList world;

        shared_ptr<Material> red =
            make_shared<Lambertian>(Color(1.0, 0.0, 0.0));

        shared_ptr<Material> green =
            make_shared<Lambertian>(Color(0.0, 1.0, 0.0));

        world.add(make_shared<Quad>(
            Point3(0, 0, 0),
            Vec3(2, 0, 0),
            Vec3(0, 2, 0),
            red
        ));

        world.add(make_shared<Quad>(
            Point3(3, 0, 0),
            Vec3(2, 0, 0),
            Vec3(0, 2, 0),
            green
        ));

        world.add(make_shared<Quad>(
            Point3(6, 0, 0),
            Vec3(2, 0, 0),
            Vec3(0, 2, 0),
            red
        ));

        world.add(make_shared<Quad>(
            Point3(9, 0, 0),
            Vec3(2, 0, 0),
            Vec3(0, 2, 0),
            green
        ));

        shared_ptr<Material> light =
        make_shared<DiffuseLight>(Color(5.0, 5.0, 5.0));

    world.add(make_shared<Quad>(
        Point3(0, 5, 0),
        Vec3(2, 0, 0),
        Vec3(0, 0, 2),
        light
    ));

        // Convert the CPU scene into GPU-compatible records.
        GPUScene cpu_scene = build_gpu_scene_from_list(world);

        std::cout << "CPU scene:\n";
        std::cout << "BVH nodes: " << cpu_scene.bvh_nodes.size() << '\n';
        std::cout << "Primitives: " << cpu_scene.primitives.size() << '\n';
        std::cout << "Materials: " << cpu_scene.materials.size() << '\n';

        // Five primitives should produce five BVH nodes.
        if (cpu_scene.bvh_nodes.size() != 5) {
            throw std::runtime_error(
                "Expected five BVH nodes for five primitives."
            );
        }

        // Upload the scene to CUDA device memory.
        GPUDeviceScene device_scene = upload_gpu_scene(cpu_scene);

        std::cout << "\nGPU scene:\n";

        std::cout << "Light primitives: "
                << cpu_scene.light_indices.size() << '\n';

        for (size_t i = 0; i < cpu_scene.light_indices.size(); ++i) {
            std::cout << "Light " << i
                    << " -> primitive "
                    << cpu_scene.light_indices[i] << '\n';
        }

        std::cout << "Uploaded light count: "
                << device_scene.light_count << '\n';

        std::cout << "BVH nodes: " << device_scene.bvh_node_count << '\n';
        std::cout << "Primitives: " << device_scene.primitive_count << '\n';
        std::cout << "Materials: " << device_scene.material_count << '\n';

        test_gpu_scene_render(device_scene);

        // Set up rays that hit each quad and one ray that misses.
        constexpr int ray_count = 5;

        GPURay rays[ray_count] = {
            {1.0f, 1.0f, -2.0f, 0.0f, 0.0f, 1.0f},
            {4.0f, 1.0f, -2.0f, 0.0f, 0.0f, 1.0f},
            {7.0f, 1.0f, -2.0f, 0.0f, 0.0f, 1.0f},
            {10.0f, 1.0f, -2.0f, 0.0f, 0.0f, 1.0f},
            {13.0f, 1.0f, -2.0f, 0.0f, 0.0f, 1.0f}
        };

        GPURay* device_bvh_rays = nullptr;
        GPUTraversalResult* device_results = nullptr;

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_bvh_rays),
                sizeof(rays)
            ),
            "Allocating device rays failed"
        );

        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_results),
                sizeof(GPUTraversalResult) * ray_count
            ),
            "Allocating device results failed"
        );

        check_cuda(
            cudaMemcpy(
                device_bvh_rays,
                rays,
                sizeof(rays),
                cudaMemcpyHostToDevice
            ),
            "Copying rays to GPU failed"
        );

        // Run one CUDA thread per ray.
        gpu_bvh_test_kernel<<<1, ray_count>>>(
            device_scene.bvh_nodes,
            device_scene.primitives,
            static_cast<int>(device_scene.bvh_node_count),
            static_cast<int>(device_scene.primitive_count),
            device_bvh_rays,
            device_results,
            ray_count
        );

        check_cuda(
            cudaGetLastError(),
            "Launching BVH test kernel failed"
        );

        check_cuda(
            cudaDeviceSynchronize(),
            "BVH test kernel execution failed"
        );

        GPUTraversalResult results[ray_count];

        check_cuda(
            cudaMemcpy(
                results,
                device_results,
                sizeof(results),
                cudaMemcpyDeviceToHost
            ),
            "Copying results from GPU failed"
        );

        // Compare GPU traversal against the CPU renderer.
        std::cout << "\nCPU vs GPU BVH intersection results:\n\n";

        int matches = 0;

        for (int i = 0; i < ray_count; i++) {
            Ray cpu_ray(
                Point3(
                    rays[i].origin_x,
                    rays[i].origin_y,
                    rays[i].origin_z
                ),
                Vec3(
                    rays[i].direction_x,
                    rays[i].direction_y,
                    rays[i].direction_z
                )
            );

            HitRecord cpu_record;

            bool cpu_hit = world.hit(
                cpu_ray,
                Interval(0.001, infinity),
                cpu_record
            );

            bool gpu_hit = results[i].hit != 0;

            bool hit_match = cpu_hit == gpu_hit;

            bool distance_match =
                !cpu_hit ||
                std::fabs(
                    static_cast<float>(cpu_record.t) - results[i].t
                ) < 1e-5f;

            bool stack_ok = results[i].stack_overflow == 0;

            bool match =
                hit_match &&
                distance_match &&
                stack_ok;

            std::cout
                << "Ray " << i
                << ": CPU = " << (cpu_hit ? "HIT " : "MISS")
                << ", GPU = " << (gpu_hit ? "HIT " : "MISS");

            if (gpu_hit) {
                std::cout
                    << ", t = " << results[i].t
                    << ", primitive = " << results[i].primitive_index;
            }

            if (results[i].stack_overflow) {
                std::cout << ", STACK OVERFLOW";
            }

            std::cout
                << (match ? "  MATCH" : "  MISMATCH")
                << '\n';

            if (match) {
                matches++;
            }
        }

        std::cout
            << "\nMatching results: "
            << matches << " / " << ray_count << '\n';

        // Release temporary GPU allocations.
        cudaFree(device_bvh_rays);
        cudaFree(device_results);

        // Release the uploaded scene.
        free_gpu_scene(device_scene);

        std::cout << "GPU memory released.\n";

        test_gpu_depth_of_field();

        return matches == ray_count ? 0 : 1;
    }
    catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}