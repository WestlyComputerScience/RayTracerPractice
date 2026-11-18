
#ifndef GPU_RENDER_CUH
#define GPU_RENDER_CUH

#include "gpu_camera.h"
#include "gpu_traversal.cuh"

// GPU-compatible pixel color.
struct GPUColor {
    float r;
    float g;
    float b;
};

// Generates a camera ray and calculates its background color.
// One CUDA thread handles one pixel.
__global__ void gpu_render_background(
    GPUCamera camera,
    GPUColor* pixels
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int pixel_count = camera.image_width * camera.image_height;

    if (index >= pixel_count) {
        return;
    }

    pixels[index] = GPUColor{
        camera.background_r,
        camera.background_g,
        camera.background_b
    };
}

// Generates camera rays and renders a scene using flat material colors.
__global__ void gpu_render_scene(
    GPUCamera camera,
    const GPUBvhNode* nodes,
    const GPUPrimitive* primitives,
    const GPUMaterial* materials,
    int node_count,
    int primitive_count,
    int material_count,
    GPUColor* pixels
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int pixel_count = camera.image_width * camera.image_height;

    if (index >= pixel_count) {
        return;
    }

    // Convert the linear index into pixel coordinates.
    int i = index % camera.image_width;
    int j = index / camera.image_width;

    // Calculate the pixel center.
    float pixel_x =
        camera.pixel00_loc.x +
        i * camera.pixel_delta_u.x +
        j * camera.pixel_delta_v.x;

    float pixel_y =
        camera.pixel00_loc.y +
        i * camera.pixel_delta_u.y +
        j * camera.pixel_delta_v.y;

    float pixel_z =
        camera.pixel00_loc.z +
        i * camera.pixel_delta_u.z +
        j * camera.pixel_delta_v.z;

    // Construct the camera ray.
    GPURay ray{
        camera.center.x,
        camera.center.y,
        camera.center.z,
        pixel_x - camera.center.x,
        pixel_y - camera.center.y,
        pixel_z - camera.center.z
    };

    // Find the closest geometry intersection.
    GPUTraversalResult hit = gpu_bvh_hit(
        nodes,
        primitives,
        node_count,
        primitive_count,
        ray,
        0.001f,
        1.0e30f
    );

    // Display traversal failures in magenta.
    if (hit.stack_overflow) {
        pixels[index] = GPUColor{1.0f, 0.0f, 1.0f};
        return;
    }

    // A ray that misses receives the background color.
    if (!hit.hit) {
        pixels[index] = GPUColor{
            camera.background_r,
            camera.background_g,
            camera.background_b
        };
        return;
    }

    // Validate the primitive index.
    if (hit.primitive_index < 0 ||
        hit.primitive_index >= primitive_count) {
        pixels[index] = GPUColor{1.0f, 0.0f, 1.0f};
        return;
    }

    const GPUPrimitive& primitive =
        primitives[hit.primitive_index];

    // Validate the material index.
    int material_index = primitive.material_index;

    if (material_index < 0 ||
        material_index >= material_count) {
        pixels[index] = GPUColor{1.0f, 0.0f, 1.0f};
        return;
    }

    // Display the material's albedo as a flat color.
    const GPUMaterial& material = materials[material_index];

    pixels[index] = GPUColor{
        material.albedo_r,
        material.albedo_g,
        material.albedo_b
    };
}

#endif
