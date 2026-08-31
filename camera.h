#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"
#include "material.h"

class Camera {
    private:
        int image_height;
        double pixel_samples_scale;
        Vec3 center;
        Vec3 pixel00_loc;
        Vec3 pixel_delta_u;
        Vec3 pixel_delta_v;
        Vec3 u;
        Vec3 v;
        Vec3 w;

        void initialize() {
            image_height = int(image_width / aspect_ratio);
            image_height = (image_height < 1) ? 1 : image_height;

            pixel_samples_scale = 1.0 / samples_per_pixel;

            center = lookfrom;

            double focal_length = (lookfrom - lookat).length();
            double theta = degrees_to_radians(vfov);
            double h = std::tan(theta/2);
            double viewport_height = 2 * h * focal_length;
            double viewport_width = viewport_height * (double(image_width)/image_height);

            w = unit_vector(lookfrom - lookat);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);

            Vec3 viewport_u = viewport_width * u;
            Vec3 viewport_v = viewport_height * -v;

            pixel_delta_u = viewport_u / image_width;
            pixel_delta_v = viewport_v / image_height;

            Vec3 viewport_upper_left = center - (focal_length * w) - viewport_u / 2 - viewport_v / 2;
            pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
        }

        Ray get_ray(int i, int j) const {
            Vec3 offset = sample_square();
            Vec3 pixel_sample = pixel00_loc + ((i + offset.x()) * pixel_delta_u) + ((j + offset.y()) * pixel_delta_v);
            Vec3 ray_origin = center;
            Vec3 ray_direction = pixel_sample - ray_origin;
            
            return Ray(ray_origin, ray_direction);
        }

        Vec3 sample_square() const {
            return Vec3(random_double() - 0.5, random_double() - 0.5, 0);
        }

        Color ray_color(const Ray& r, int depth, const Hittable& world) const {
            if (depth <= 0) return Color (0, 0, 0);

            HitRecord rec;
            if (world.hit(r, Interval(0.001, infinity), rec)) { // 0.001 solves shadow acne problem
                Ray scattered;
                Color attenuation;
                if (rec.mat->scatter(r, rec, attenuation, scattered)) {
                    return attenuation * ray_color(scattered, depth - 1, world);
                }
                return Color(0, 0, 0);
            }

            Vec3 unit_direction = unit_vector(r.direction());
            double a = 0.5 * (unit_direction.y() + 1.0);
            return (1.0 - a) * Color(1.0, 1.0, 1.0) + a * Color(0.5, 0.7, 1.0);
        }

    public:
        double aspect_ratio = 1.0;
        int image_width = 100;
        int samples_per_pixel = 10;
        int max_ray_bounces = 10;
        double vfov = 90; // vertical field of view
        Vec3 lookfrom = Vec3(0, 0, 0);
        Vec3 lookat = Vec3(0, 0, -1);
        Vec3 vup = Vec3(0, 1, 0);

        void render(const Hittable& world) {
            initialize();

            std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

            for (int j = 0; j < image_height; j++) {
                std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;

                for (int i = 0; i < image_width; i++) {
                    Color pixel_color(0, 0, 0);
                    for (int sample = 0; sample < samples_per_pixel; sample++) {
                        Ray r = get_ray(i, j);
                        pixel_color += ray_color(r, max_ray_bounces, world);
                    }
                    write_color(std::cout, pixel_samples_scale * pixel_color);
                }
            }
            std::clog << "\rDone.                 \n";
        }
};

#endif