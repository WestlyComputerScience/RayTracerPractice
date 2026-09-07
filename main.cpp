#include "common_constants.h"
#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include "bvh.h"

void make_ray_tracing_in_one_weekend_cover(HittableList& world) {
    shared_ptr<Material> material_ground = make_shared<Lambertian>(Color(0.5, 0.5, 0.5));
    world.add(make_shared<Sphere>(Point3(0, -1000, 0), 1000, material_ground));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            double choose_mat = random_double();
            Point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

            if ((center - Point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<Material> sphere_material;

                if (choose_mat < 0.1) {
                    shared_ptr<Material> sphere_material_1 = make_shared<Dielectric>(1.5);
                    shared_ptr<Material> sphere_material_2 = make_shared<Dielectric>(1.00 / 1.5);
                    world.add(make_shared<Sphere>(center, 0.2, sphere_material_1));
                    world.add(make_shared<Sphere>(center, 0.15, sphere_material_1));
                } else if (choose_mat < 0.8) {
                    // diffuse
                    Color albedo = Color::random() * Color::random();
                    sphere_material = make_shared<Lambertian>(albedo);
                    world.add(make_shared<Sphere>(center, 0.2, sphere_material));
                    Point3 center2 = center + Vec3(0, random_double(0, 0.5), 0);
                    world.add(make_shared<Sphere>(center, center2, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    Color albedo = Color::random(0.5, 1);
                    double fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<Metal>(albedo, fuzz);
                    world.add(make_shared<Sphere>(center, 0.2, sphere_material));
                } else {
                    sphere_material = make_shared<Dielectric>(1.5);
                    world.add(make_shared<Sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }
}

int main() {
    HittableList world;

    make_ray_tracing_in_one_weekend_cover(world);

    world = HittableList(make_shared<BvhNode>(world));

    Camera cam;

    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = 400;
    cam.samples_per_pixel = 100;
    cam.max_ray_bounces = 50;

    cam.vfov = 20;
    cam.lookfrom = Point3(13,2,3);
    cam.lookat = Point3(0,0,0);
    cam.vup = Vec3(0,1,0);

    cam.defocus_angle = 0.6;
    cam.focus_dist = 10.0;

    cam.render(world);
}

/* === Archived tests ===

shared_ptr<Material> material_ground = make_shared<Lambertian>(Color(0.8, 0.8, 0.0));
shared_ptr<Material> material_center = make_shared<Lambertian>(Color(0.1, 0.2, 0.5));

// shared_ptr<Material> material_left = make_shared<Dielectric>(1.00 / 1.33); // refraction of air / refraction of water

// hollow glass sphere with air inside
// shared_ptr<Material> material_left = make_shared<Dielectric>(1.5);
// shared_ptr<Material> material_bubble = make_shared<Dielectric>(1.00 / 1.5);

shared_ptr<Material> material_left = make_shared<Dielectric>(1.5);
shared_ptr<Material> material_bubble = make_shared<Dielectric>(1.00 / 1.5);
shared_ptr<Material> material_right = make_shared<Metal>(Color(0.8, 0.6, 0.2), 1.0);

world.add(make_shared<Sphere>(Vec3(0.0, -100.5, -1.0), 100.0, material_ground));
world.add(make_shared<Sphere>(Vec3(0.0, 0.0, -1.2), 0.5, material_center));
world.add(make_shared<Sphere>(Vec3(-1.0, 0.0, -1.0), 0.5, material_left));
world.add(make_shared<Sphere>(Vec3(-1.0, 0.0, -1.0), 0.4, material_bubble));
world.add(make_shared<Sphere>(Vec3(1.0, 0.0, -1.0), 0.5, material_right));

*/