#ifndef GPU_CAMERA_BUILDER_H
#define GPU_CAMERA_BUILDER_H

#include "camera.h"
#include "gpu_camera.h"

inline GPUVec3 convert_vec3_to_gpu(const Vec3& v) {
    return GPUVec3{
        static_cast<float>(v.x()),
        static_cast<float>(v.y()),
        static_cast<float>(v.z())
    };
}

inline GPUCamera build_gpu_camera(const Camera& camera) {
    GPUCamera gpu_camera{};

    gpu_camera.center =
        convert_vec3_to_gpu(camera.get_center());

    gpu_camera.pixel00_loc =
        convert_vec3_to_gpu(camera.get_pixel00_loc());

    gpu_camera.pixel_delta_u =
        convert_vec3_to_gpu(camera.get_pixel_delta_u());

    gpu_camera.pixel_delta_v =
        convert_vec3_to_gpu(camera.get_pixel_delta_v());

    gpu_camera.defocus_disk_u =
        convert_vec3_to_gpu(camera.get_defocus_disk_u());

    gpu_camera.defocus_disk_v =
        convert_vec3_to_gpu(camera.get_defocus_disk_v());

    gpu_camera.defocus_angle =
        static_cast<float>(camera.config.defocus_angle);

    gpu_camera.image_width = camera.config.image_width;
    gpu_camera.image_height = camera.get_image_height();

    gpu_camera.samples_per_pixel =
        camera.config.samples_per_pixel;

    gpu_camera.max_ray_bounces =
        camera.config.max_ray_bounces;

    gpu_camera.background_r =
        static_cast<float>(camera.config.background.r());

    gpu_camera.background_g =
        static_cast<float>(camera.config.background.g());

    gpu_camera.background_b =
        static_cast<float>(camera.config.background.b());

    return gpu_camera;
}

#endif