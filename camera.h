#ifndef CAMERA_H
#define CAMERA_H

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb_image_write.h" // allows hdr images to be written
#include <iostream>

#include <omp.h>
#include "hittable.h"
#include "pdf.h"
#include "material.h"
#include "cube_map.h"
#include "rtw_stb_image.h"
#include "camera_config.h"

/**
* Manages image generation and streams PPM image data directly to standard output.
*/
class Camera {
    private:
        int image_height;
        real pixel_samples_scale; // used at the end of rendering to calculate averahe accumulated color samples
        int sqrt_spp; // square root fo samples per pixel
        real recip_sqrt_spp; // 1 / spp
        Point3 center; // cameras focal origin point in world space
        Point3 pixel00_loc; // 3D space coord corresponding to the center of the top pixel (0, 0)
        Vec3 pixel_delta_u; // 3D displacement vectors u, v, between 2 adjacent horizontal/verticle vectors
        Vec3 pixel_delta_v;
        Vec3 u, v, w; // vectors pointing backward, directly, and directly up relative to camera's orientation
        Vec3 defocus_disk_u; // horizontal/verticle radius vector scaling unit circles samples to match a physical len's aperture
        Vec3 defocus_disk_v;
        CubeMap skybox; // cube mapping

        /**
        * Constructs a jittered ray targeting pixel location (i, j)
        */
        Ray get_ray(int i, int j, int s_i, int s_j) const {
            Vec3 offset = sample_square_stratified(s_i, s_j);

            Point3 pixel_sample =
                pixel00_loc
                + ((i + offset.x()) * pixel_delta_u)
                + ((j + offset.y()) * pixel_delta_v);

            Point3 ray_origin =
                (config.defocus_angle <= 0)
                ? center
                : defocus_disk_sample();

            Vec3 ray_direction = pixel_sample - ray_origin;

            real ray_time = random_real();

            return Ray(ray_origin, ray_direction, ray_time);
        }

        /**
        * Returns the vector to a random point in the square sub-pixel specified by grid indicies s_i s_j
        */
        Vec3 sample_square_stratified(int s_i, int s_j) const {
            real px = ((s_i + random_real()) * recip_sqrt_spp) - 0.5;
            real py = ((s_j + random_real()) * recip_sqrt_spp) - 0.5;

            return Vec3(px, py, 0);
        }

        /**
        * Provides a uniform 2D random offset inside a square centered at (0, 0) with
        * side lengths (-0.5, 0.5)
        */
        Vec3 sample_square() const {
            return Vec3(random_real() - 0.5, random_real() - 0.5, 0);
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
        Color ray_color(
            const Ray& r,
            int depth,
            const Hittable& world,
            const Hittable& lights
        ) const {
            if (depth <= 0) return Color(0, 0, 0);

            HitRecord rec;

            if (!world.hit(r, Interval(0.001, infinity), rec))
                return config.background;

            ScatterRecord srec;

            Color color_from_emission =
                rec.mat->emitted(r, rec, rec.u, rec.v, rec.p);

            if (!rec.mat->scatter(r, rec, srec))
                return color_from_emission;

            if (srec.skip_pdf) {
                HitRecord secondary_rec;

                if (world.hit(
                    srec.skip_pdf_ray,
                    Interval(0.001, infinity),
                    secondary_rec
                )) {
                    return secondary_rec.mat->emitted(
                        srec.skip_pdf_ray,
                        secondary_rec,
                        secondary_rec.u,
                        secondary_rec.v,
                        secondary_rec.p
                    );
                }

                return config.background;
            }

            shared_ptr<Pdf> light_ptr =
                make_shared<HittablePdf>(lights, rec.p);

            MixturePdf p(light_ptr, srec.pdf_ptr);

            Ray scattered(rec.p, p.generate(), r.time());

            real pdf_value = p.value(scattered.direction());

            if (pdf_value <= 1e-8)
                return color_from_emission;

            Color sample_color =
                ray_color(scattered, depth - 1, world, lights);

            Color scattering_value = rec.mat->scattering_value(
                r,
                rec,
                scattered,
                srec.attenuation
            );

            return color_from_emission
                + scattering_value * (1.0 / pdf_value) * sample_color;
        }
    public:
        CameraConfig config;
        real& aspect_ratio = config.aspect_ratio;
        int& image_width = config.image_width;
        int& samples_per_pixel = config.samples_per_pixel;
        int& max_ray_bounces = config.max_ray_bounces;
        real& vfov = config.vfov;
        real& defocus_angle = config.defocus_angle;
        real& focus_dist = config.focus_dist;
        Point3& lookfrom = config.lookfrom;
        Point3& lookat = config.lookat;
        Vec3& vup = config.vup;
        Color& background = config.background;

        Point3 get_center() const { return center; }
        Point3 get_pixel00_loc() const { return pixel00_loc; }

        Vec3 get_pixel_delta_u() const { return pixel_delta_u; }
        Vec3 get_pixel_delta_v() const { return pixel_delta_v; }

        Vec3 get_defocus_disk_u() const { return defocus_disk_u; }
        Vec3 get_defocus_disk_v() const { return defocus_disk_v; }

        int get_image_height() const { return image_height; }

        /**
        * Computes frame params before rendering.
        */
        void initialize() {
            // Image height.
            image_height = int(config.image_width / config.aspect_ratio);
            image_height = (image_height < 1) ? 1 : image_height;

            sqrt_spp = int(std::sqrt(config.samples_per_pixel));
            recip_sqrt_spp = 1.0 / sqrt_spp;

            int actual_samples = sqrt_spp * sqrt_spp;
            pixel_samples_scale = 1.0 / actual_samples;

            center = config.lookfrom;

            // Viewport dimensioning.
            real theta = degrees_to_radians(config.vfov);
            real h = std::tan(theta / 2);
            real viewport_height = 2 * h * config.focus_dist;
            real viewport_width =
                viewport_height * (real(config.image_width) / image_height);

            // Camera orientation vectors.
            w = unit_vector(config.lookfrom - config.lookat);
            u = unit_vector(cross(config.vup, w));
            v = cross(w, u);

            Vec3 viewport_u = viewport_width * u;
            Vec3 viewport_v = viewport_height * -v;

            pixel_delta_u = viewport_u / config.image_width;
            pixel_delta_v = viewport_v / image_height;

            // Upper-left pixel location.
            Point3 viewport_upper_left =
                center
                - (config.focus_dist * w)
                - viewport_u / 2
                - viewport_v / 2;

            pixel00_loc =
                viewport_upper_left
                + 0.5 * (pixel_delta_u + pixel_delta_v);

            // Thin-lens aperture.
            real defocus_radius =
                config.focus_dist *
                std::tan(degrees_to_radians(config.defocus_angle / 2));

            defocus_disk_u = u * defocus_radius;
            defocus_disk_v = v * defocus_radius;

            skybox = CubeMap(
                "images/cubeMapping/coolHouse/right.hdr",
                "images/cubeMapping/coolHouse/left.hdr",
                "images/cubeMapping/coolHouse/up.hdr",
                "images/cubeMapping/coolHouse/down.hdr",
                "images/cubeMapping/coolHouse/front.hdr",
                "images/cubeMapping/coolHouse/back.hdr"
            );
        }

        /**
        * Main entry point executing scene rendering and PPM output.
        */
        void render(const Hittable& world, const Hittable& lights) {
            initialize();
            render_initialized(world, lights);
        }

        void render_initialized(const Hittable& world, const Hittable& lights) {
            omp_set_num_threads(14);

            std::vector<std::vector<Color>> framebuffer(
                image_height,
                std::vector<Color>(config.image_width)
            );

            #pragma omp parallel for schedule(dynamic, 1)
            for (int j = 0; j < image_height; j++) {
                #pragma omp critical
                {
                    std::clog
                        << "\rScanlines remaining: "
                        << (image_height - j)
                        << ' '
                        << std::flush;
                }

                for (int i = 0; i < config.image_width; i++) {
                    Color pixel_color(0, 0, 0);

                    for (int s_j = 0; s_j < sqrt_spp; s_j++) {
                        for (int s_i = 0; s_i < sqrt_spp; s_i++) {
                            Ray r = get_ray(i, j, s_i, s_j);

                            pixel_color += ray_color(
                                r,
                                config.max_ray_bounces,
                                world,
                                lights
                            );
                        }
                    }

                    framebuffer[j][i] = pixel_samples_scale * pixel_color;
                }
            }

            std::vector<float> hdr_pixels(
                config.image_width * image_height * 3
            );

            int index = 0;

            for (int j = 0; j < image_height; j++) {
                for (int i = 0; i < config.image_width; i++) {
                    hdr_pixels[index++] = framebuffer[j][i].r();
                    hdr_pixels[index++] = framebuffer[j][i].g();
                    hdr_pixels[index++] = framebuffer[j][i].b();
                }
            }

            const char* output_file = "cool.hdr";

            if (stbi_write_hdr(
                output_file,
                config.image_width,
                image_height,
                3,
                hdr_pixels.data()
            )) {
                std::clog
                    << "\nSuccessfully wrote HDR image to "
                    << output_file << "\n";
            } else {
                std::cerr << "\nERROR: Failed to write HDR image file.\n";
            }

            std::clog << "Done.                \n";
        }
};

#endif