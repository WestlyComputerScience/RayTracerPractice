#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"
#include "material.h"

/**
* Manages image generation and streams PPM image data directly to standard output.
*/
class Camera {
    private:
        int image_height;
        double pixel_samples_scale; // used at the end of rendering to calculate averahe accumulated color samples
        Point3 center; // cameras focal origin point in world space
        Point3 pixel00_loc; // 3D space coord corresponding to the center of the top pixel (0, 0)
        Vec3 pixel_delta_u; // 3D displacement vectors u, v, between 2 adjacent horizontal/verticle vectors
        Vec3 pixel_delta_v;
        Vec3 u, v, w; // vectors pointing backward, directly, and directly up relative to camera's orientation
        Vec3 defocus_disk_u; // horizontal/verticle radius vector scaling unit circles samples to match a physical len's aperture
        Vec3 defocus_disk_v;

        /**
        * Computes frame params before rendering.
        */
        void initialize() {
            // image height
            image_height = int(image_width / aspect_ratio);
            image_height = (image_height < 1) ? 1 : image_height;

            pixel_samples_scale = 1.0 / samples_per_pixel;

            center = lookfrom;

            // veiwpoint dimensioning
            double theta = degrees_to_radians(vfov);
            double h = std::tan(theta/2);
            double viewport_height = 2 * h * focus_dist;
            double viewport_width = viewport_height * (double(image_width)/image_height);

            // camera orientation vectors
            w = unit_vector(lookfrom - lookat);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);

            Vec3 viewport_u = viewport_width * u;
            Vec3 viewport_v = viewport_height * -v;

            pixel_delta_u = viewport_u / image_width;
            pixel_delta_v = viewport_v / image_height;

            // finding the upper left corner
            Point3 viewport_upper_left = center - (focus_dist * w) - viewport_u / 2 - viewport_v / 2;
            pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

            // thin-lens aperture
            double defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
            defocus_disk_u = u * defocus_radius;
            defocus_disk_v = v * defocus_radius;
        }

        /**
        * Constructs a jittered ray targeting pixel location (i, j)
        */
        Ray get_ray(int i, int j) const {
            Vec3 offset = sample_square();
            Point3 pixel_sample = pixel00_loc + ((i + offset.x()) * pixel_delta_u) + ((j + offset.y()) * pixel_delta_v);
            Point3 ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample();
            Vec3 ray_direction = pixel_sample - ray_origin;
            double ray_time = random_double();
            return Ray(ray_origin, ray_direction, ray_time);
        }

        /**
        * Provides a uniform 2D random offset inside a square centered at (0, 0) with
        * side lengths (-0.5, 0.5)
        */
        Vec3 sample_square() const {
            return Vec3(random_double() - 0.5, random_double() - 0.5, 0);
        }

        /**
        * Simulates a physical camera lens aperture by sampling a point on a disk.
        */
        Point3 defocus_disk_sample() const {
            Vec3 p = random_in_unit_disk();
            return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
        }

        /**
        * Evaluates incoming radiance for a ray through recursive path tracing.
        */
        Color ray_color(const Ray& r, int depth, const Hittable& world) const {
            if (depth <= 0) return Color (0, 0, 0); // returns black if exceeds depth

            // returns black if no intersection
            HitRecord rec;
            if (!world.hit(r, Interval(0.001, infinity), rec)) { // 0.001 solves shadow acne problem
                return background;
            }

            // Samples light emissions from emissive materials
            Ray scattered;
            Color attenuation;
            Color color_from_emission = rec.mat->emitted(rec.u, rec.v, rec.p);

            // if the ray is absorbed or non-scattering, returns the color from emission
            if (!rec.mat->scatter(r, rec, attenuation, scattered)) return color_from_emission;

            // recursive radiance equation
            Color color_from_scatter = attenuation * ray_color(scattered, depth - 1, world);
            
            return color_from_emission + color_from_scatter;
        }

    public:
        double aspect_ratio = 1.0; // ratio of image width to height
        int image_width = 100;
        int samples_per_pixel = 10; // num of randomized antialiasing ray samples per pixel
        int max_ray_bounces = 10; // maximum allowed depth for recursive bounce paths
        double vfov = 90; // vertical field of view
        double defocus_angle = 0; // disk aperture angle for depth-of-field
        double focus_dist = 10; // distance from lens origin to focus plane
        Point3 lookfrom = Point3(0, 0, 0); // camera position to world coords
        Point3 lookat = Point3(0, 0, -1); // target point camera looks toward
        Vec3 vup = Vec3(0, 1, 0); // upward orientation reference vector
        Color background; // base color returned when a ray misses scene objects

        /**
        * Main entry point executing scene rendering and PPM output.
        */
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