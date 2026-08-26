#include "common_constants.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"

Color ray_color(const Ray& r, const Hittable& world) {
    HitRecord rec;
    if (world.hit(r, 0, infinity, rec)) {
        return 0.5 * (rec.normal + Color(1, 1, 1));
    }

    Vec3 unit_direction = unit_vector(r.direction()); // -1 to 1
    double a = 0.5 * (unit_direction.y() + 1.0); // 0 to 1
    return (1.0 - a) * Color (1.0, 1.0, 1.0) + a*Color(0.1, 0.5, 0.7);
}

int main() {
    double aspect_ratio = 16.0 / 9.0;
    int image_width = 400;

    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    HittableList world;
    world.add(make_shared<Sphere>(Vec3(0, 0, -1), 0.5));
    world.add(make_shared<Sphere>(Vec3(0, -100.5, -1), 100));

    double focal_length = 1.0;
    double viewport_height = 2.0;
    double viewport_width = viewport_height * (double(image_width) / image_height);
    Vec3 camera_center = Vec3(0, 0, 0);

    Vec3 viewport_u = Vec3(viewport_width, 0, 0);
    Vec3 viewport_v = Vec3(0, -viewport_height, 0);

    Vec3 pixel_delta_u = viewport_u / image_width;
    Vec3 pixel_delta_v = viewport_v / image_height;

    Vec3 viewport_upper_left = camera_center - Vec3(0, 0, focal_length) - viewport_u / 2 - viewport_v / 2;
    Vec3 pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    for (int j = 0; j < image_height; j++) {
        std::clog << "\rScanlines remaining: " << (image_height - j) << ' ' << std::flush;
        for (int i = 0; i < image_width; i++) {
            Vec3 pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
            Vec3 ray_direction = pixel_center - camera_center;
            Ray r(camera_center, ray_direction);

            Color pixel_color = ray_color(r, world);
            write_color(std::cout, pixel_color);
        }
    }
    std::clog << "\rDone.                 \n";
}