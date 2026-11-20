#ifndef GPU_CAMERA_H
#define GPU_CAMERA_H

struct GPUVec3 {
    float x;
    float y;
    float z;
};

struct GPUCamera {
    GPUVec3 center;

    GPUVec3 pixel00_loc;
    GPUVec3 pixel_delta_u;
    GPUVec3 pixel_delta_v;

    GPUVec3 defocus_disk_u;
    GPUVec3 defocus_disk_v;

    float defocus_angle;

    int image_width;
    int image_height;

    int samples_per_pixel;
    int max_ray_bounces;
    int sqrt_spp;
    float recip_sqrt_spp;

    float background_r;
    float background_g;
    float background_b;
};

#endif