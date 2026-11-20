
#ifndef GPU_CAMERA_CUH
#define GPU_CAMERA_CUH

#include "gpu_camera.h"
#include "gpu_intersection.cuh"
#include "gpu_random.cuh"

__device__ inline void gpu_sample_square_stratified(
    const GPUCamera& camera,
    int s_i,
    int s_j,
    unsigned int& seed,
    float& offset_x,
    float& offset_y
) {
    float random_x = gpu_random_float(seed);
    float random_y = gpu_random_float(seed);

    offset_x =
        ((s_i + random_x) * camera.recip_sqrt_spp) - 0.5f;

    offset_y =
        ((s_j + random_y) * camera.recip_sqrt_spp) - 0.5f;
}

__device__ inline void gpu_random_in_unit_disk(
    unsigned int& seed,
    float& x,
    float& y
) {
    while (true) {
        x = 2.0f * gpu_random_float(seed) - 1.0f;
        y = 2.0f * gpu_random_float(seed) - 1.0f;

        if (x * x + y * y < 1.0f) {
            return;
        }
    }
}


__device__ inline GPUVec3 gpu_defocus_disk_sample(
    const GPUCamera& camera,
    unsigned int& seed
) {
    float x;
    float y;

    gpu_random_in_unit_disk(seed, x, y);

    return GPUVec3{
        camera.center.x +
            x * camera.defocus_disk_u.x +
            y * camera.defocus_disk_v.x,

        camera.center.y +
            x * camera.defocus_disk_u.y +
            y * camera.defocus_disk_v.y,

        camera.center.z +
            x * camera.defocus_disk_u.z +
            y * camera.defocus_disk_v.z
    };
}

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


__global__ void gpu_generate_camera_sample_rays(
    GPUCamera camera,
    GPURay* rays
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int pixel_count =
        camera.image_width * camera.image_height;

    int sample_count_per_pixel =
        camera.sqrt_spp * camera.sqrt_spp;

    if (sample_count_per_pixel <= 0) {
        return;
    }

    int total_sample_count =
        pixel_count * sample_count_per_pixel;

    if (index >= total_sample_count) {
        return;
    }

    // Determine which pixel this sample belongs to.
    int pixel_index = index / sample_count_per_pixel;

    int i = pixel_index % camera.image_width;
    int j = pixel_index / camera.image_width;

    // Determine the sample's position in the stratified grid.
    int sample_index = index % sample_count_per_pixel;

    int s_i = sample_index % camera.sqrt_spp;
    int s_j = sample_index / camera.sqrt_spp;

    // Give each sample its own deterministic random seed.
    unsigned int seed =
        static_cast<unsigned int>(index) + 1u;

    // Generate the stratified pixel offset.
    float offset_x;
    float offset_y;

    gpu_sample_square_stratified(
        camera,
        s_i,
        s_j,
        seed,
        offset_x,
        offset_y
    );

    // Calculate the sampled pixel location.
    float pixel_x =
        camera.pixel00_loc.x +
        (i + offset_x) * camera.pixel_delta_u.x +
        (j + offset_y) * camera.pixel_delta_v.x;

    float pixel_y =
        camera.pixel00_loc.y +
        (i + offset_x) * camera.pixel_delta_u.y +
        (j + offset_y) * camera.pixel_delta_v.y;

    float pixel_z =
        camera.pixel00_loc.z +
        (i + offset_x) * camera.pixel_delta_u.z +
        (j + offset_y) * camera.pixel_delta_v.z;

    // Determine the ray origin.
    GPUVec3 ray_origin = camera.center;

    if (camera.defocus_angle > 0.0f) {
        ray_origin = gpu_defocus_disk_sample(
            camera,
            seed
        );
    }

    // Calculate the ray direction.
    float direction_x = pixel_x - ray_origin.x;
    float direction_y = pixel_y - ray_origin.y;
    float direction_z = pixel_z - ray_origin.z;

    // Store the generated ray.
    rays[index] = GPURay{
        ray_origin.x,
        ray_origin.y,
        ray_origin.z,
        direction_x,
        direction_y,
        direction_z
    };
}

#endif
