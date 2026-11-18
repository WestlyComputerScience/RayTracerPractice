
#ifndef GPU_CAMERA_CUH
#define GPU_CAMERA_CUH

#include "gpu_camera.h"
#include "gpu_intersection.cuh"

// Generates one center-of-pixel ray per pixel.
__global__ void gpu_generate_camera_rays(
    GPUCamera camera,
    GPURay* rays
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int pixel_count = camera.image_width * camera.image_height;

    if (index >= pixel_count) {
        return;
    }

    // Convert the linear index into pixel coordinates.
    int i = index % camera.image_width;
    int j = index / camera.image_width;

    // Calculate the center of this pixel.
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

    // Calculate the ray direction.
    float direction_x = pixel_x - camera.center.x;
    float direction_y = pixel_y - camera.center.y;
    float direction_z = pixel_z - camera.center.z;

    // Store the generated ray.
    rays[index] = GPURay{
        camera.center.x,
        camera.center.y,
        camera.center.z,
        direction_x,
        direction_y,
        direction_z
    };
}

#endif
