#ifndef CAMERA_CONFIG_H
#define CAMERA_CONFIG_H

#include "common_constants.h"

struct CameraConfig {
    int image_width = 100;

    real aspect_ratio = 1.0;
    int samples_per_pixel = 10;
    int max_ray_bounces = 10;

    real vfov = 90;
    real defocus_angle = 0;
    real focus_dist = 10;

    Point3 lookfrom = Point3(0, 0, 0);
    Point3 lookat = Point3(0, 0, -1);
    Vec3 vup = Vec3(0, 1, 0);

    Color background = Color(0, 0, 0);
};

#endif