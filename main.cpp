#define TINYOBJLOADER_IMPLEMENTATION

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "external/tiny_obj_loader.h"
#pragma GCC diagnostic pop

#include "external/tiny_obj_loader.h"

#include "common_constants.h"
#include "camera.h"
#include "constant_medium.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
#include "bvh.h"
#include "triangle.h"
#include "quad.h"

/**
* The main class used to setup ray tracer scenarios.
*/

void set_camera(HittableList& world, bool isDefaultLight) {
    Camera cam;

    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = 400;
    cam.samples_per_pixel = 100;
    cam.max_ray_bounces = 50;
    if (isDefaultLight) {
        cam.background = Color(0.7, 0.8, 1.0);
    } else {
        cam.background = Color(0, 0, 0);
    }

    cam.vfov = 20;
    cam.lookfrom = Point3(26, 3, 6);
    cam.lookat = Point3(0,2,0);
    cam.vup = Vec3(0,1,0);

    cam.defocus_angle = 0;
    // cam.focus_dist = 10.0;

    cam.render(world);
}

void make_ray_tracing_in_one_weekend_cover(HittableList& world) {
    // shared_ptr<Material> material_ground = make_shared<Lambertian>(Color(0.5, 0.5, 0.5));
    // world.add(make_shared<Sphere>(Point3(0, -1000, 0), 1000, material_ground));

    shared_ptr<Texture> checker = make_shared<CheckerTexture>(0.32, Color(0.2, 0.3, 0.1), Color(0.9, 0.9, 0.9));
    world.add(make_shared<Sphere>(Point3(0, -1000, 0), 1000, make_shared<Lambertian>(checker)));

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

void bouncing_spheres() {
    HittableList world;

    make_ray_tracing_in_one_weekend_cover(world);

    world = HittableList(make_shared<BvhNode>(world));

    set_camera(world, true);
}

void checkered_spheres() {
    HittableList world;

    shared_ptr<Texture> checker = make_shared<CheckerTexture>(0.32, Color(0.2, 0.3, 0.1), Color(0.9, 0.9, 0.9));

    world.add(make_shared<Sphere>(Point3(0, -10, 0), 10, make_shared<Lambertian>(checker)));
    world.add(make_shared<Sphere>(Point3(0, 10, 0), 10, make_shared<Lambertian>(checker)));

    set_camera(world, true);
}

void moon() { 
    HittableList world;

    // shared_ptr<Texture> earth_texture = make_shared<ImageTexture>("earthmap.jpg");
    shared_ptr<Texture> earth_texture = make_shared<ImageTexture>("moon.jpg");
    shared_ptr<Material> earth_surface = make_shared<Lambertian>(earth_texture);
    shared_ptr<Hittable> globe = make_shared<Sphere>(Point3(0, 0, 0), 2, earth_surface);
    world.add(globe);

    set_camera(world, true);
}

void perlin_spheres() {
    HittableList world;

    shared_ptr<Texture> pertext = make_shared<NoiseTexture>(4);
    world.add(make_shared<Sphere>(Point3(0, -1000, 0), 1000, make_shared<Lambertian>(pertext)));
    world.add(make_shared<Sphere>(Point3(0, 2, 0), 2, make_shared<Lambertian>(pertext)));

    set_camera(world, true);
}

void triangles() {
    HittableList world;

    shared_ptr<Texture> pertext = make_shared<NoiseTexture>(4);
    world.add(make_shared<Sphere>(Point3(0, -1000, 0), 1000, make_shared<Lambertian>(pertext)));

    shared_ptr<Texture> triangleTex = make_shared<SolidColor>(0.15, 0.6, 0.8);
    world.add(make_shared<Triangle>(Point3(2, 0, 0), Point3(2, 0, 2), Point3(2, 1, 1), make_shared<Lambertian>(triangleTex)));

    set_camera(world, true);
}

void hat() {
    HittableList world;

    shared_ptr<Texture> sphereTex = make_shared<SolidColor>(0.2, 0.6, 0.8);
    world.add(make_shared<Sphere>(Point3(3, 0, 2), 1, make_shared<Lambertian>(sphereTex)));
    world.add(make_shared<Sphere>(Point3(3, 0, -2), 1, make_shared<Lambertian>(sphereTex)));

    shared_ptr<Texture> triangleTex = make_shared<SolidColor>(0.15, 0.6, 0.8);
    world.add(make_shared<Triangle>(Point3(2, 0, 0), Point3(3, 0, 2), Point3(3, 4, 0), make_shared<Lambertian>(triangleTex)));
    world.add(make_shared<Triangle>(Point3(3, 0, 2), Point3(4, 0, 0), Point3(3, 4, 0), make_shared<Lambertian>(triangleTex)));
    world.add(make_shared<Triangle>(Point3(4, 0, 0), Point3(3, 0, -2), Point3(3, 4, 0), make_shared<Lambertian>(triangleTex)));
    world.add(make_shared<Triangle>(Point3(3, 0, -2), Point3(2, 0, 0), Point3(3, 4, 0), make_shared<Lambertian>(triangleTex)));

    set_camera(world, true);
}

void poop_emoji() {
    HittableList world;

    shared_ptr<Texture> poopTex = make_shared<ImageTexture>("poop_emoji.jpg");
    world.add(make_shared<Triangle>(Point3(2, 0, -3), Point3(2, 0, 5), Point3(2, 3, 1), make_shared<Lambertian>(poopTex)));

    set_camera(world, true);
}

void quads() {
    HittableList world;

    shared_ptr<Material> left_red = make_shared<Lambertian>(Color(1.0, 0.2, 0.2));
    shared_ptr<Material> back_green = make_shared<Lambertian>(Color(0.2, 1.0, 0.2));
    shared_ptr<Material> right_blue = make_shared<Lambertian>(Color(0.2, 0.2, 1.0));
    shared_ptr<Material> upper_orange = make_shared<Lambertian>(Color(1.0, 0.5, 0.0));
    shared_ptr<Material> lower_teal = make_shared<Lambertian>(Color(0.2, 0.8, 0.8));

    world.add(make_shared<Quad>(Point3(-3, -2, 5), Vec3(0, 0, -4), Vec3(0, 4, 0), left_red));
    world.add(make_shared<Quad>(Point3(-2, -2, 0), Vec3(4, 0, 0), Vec3(0, 4, 0), back_green));
    world.add(make_shared<Quad>(Point3(3, -2, 1), Vec3(0, 0, 4), Vec3(0, 4, 0), right_blue));
    world.add(make_shared<Quad>(Point3(-2, 3, 1), Vec3(4, 0, 0), Vec3(0, 0, 4), upper_orange));
    world.add(make_shared<Quad>(Point3(-2, -3, 5), Vec3(4, 0, 0), Vec3(0, 0, -4), lower_teal));

    set_camera(world, true);
}

void simple_light() {
    HittableList world;

    shared_ptr<Texture> pertext = make_shared<NoiseTexture>(4);
    world.add(make_shared<Sphere>(Point3(0,-1000,0), 1000, make_shared<Lambertian>(pertext)));
    world.add(make_shared<Sphere>(Point3(0,2,0), 2, make_shared<Lambertian>(pertext)));

    shared_ptr<Material> difflight = make_shared<DiffuseLight>(Color(4,4,4));
    // world.add(make_shared<Sphere>(Point3(0, 7, 0), 2, difflight));
    world.add(make_shared<Quad>(Point3(3,1,-2), Vec3(2,0,0), Vec3(0,2,0), difflight));

    Camera cam;
    cam.aspect_ratio      = 16.0 / 9.0;
    cam.image_width       = 400;
    cam.samples_per_pixel = 100;
    cam.max_ray_bounces   = 50;
    cam.background        = Color(0, 0, 0);

    cam.vfov     = 20;
    cam.lookfrom = Point3(26, 3, 6);
    cam.lookat   = Point3(0, 2, 0);
    cam.vup      = Vec3(0, 1, 0);
    cam.defocus_angle = 0;

    cam.render(world);
}

void cornell_box() {
    HittableList world;

    shared_ptr<Material> red = make_shared<Lambertian>(Color(0.65, 0.05, 0.05));
    shared_ptr<Material> white = make_shared<Lambertian>(Color(0.73, 0.73, 0.73));
    shared_ptr<Material> green = make_shared<Lambertian>(Color(0.12, 0.45, 0.15));
    shared_ptr<Material> light = make_shared<DiffuseLight>(Color(15, 15, 15));

    world.add(make_shared<Quad>(Point3(555,0,0), Vec3(0,555,0), Vec3(0,0,555), green));
    world.add(make_shared<Quad>(Point3(0,0,0), Vec3(0,555,0), Vec3(0,0,555), red));
    world.add(make_shared<Quad>(Point3(343, 554, 332), Vec3(-130,0,0), Vec3(0,0,-105), light));
    world.add(make_shared<Quad>(Point3(0,0,0), Vec3(555,0,0), Vec3(0,0,555), white));
    world.add(make_shared<Quad>(Point3(555,555,555), Vec3(-555,0,0), Vec3(0,0,-555), white));
    world.add(make_shared<Quad>(Point3(0,0,555), Vec3(555,0,0), Vec3(0,555,0), white));

    world.add(box(Point3(130, 0, 65), Point3(295, 165, 230), white));
    world.add(box(Point3(265, 0, 295), Point3(430, 330, 460), white));

    shared_ptr<Hittable> box1 = box(Point3(0, 0, 0), Point3(165, 330, 165), white);
    box1 = make_shared<RotateY>(box1, 15);
    box1 = make_shared<Translate>(box1, Vec3(265, 0, 295));
    world.add(box1);

    shared_ptr<Hittable> box2 = box(Point3(0, 0, 0), Point3(165, 165, 165), white);
    box2 = make_shared<RotateY>(box2, -18);
    box2 = make_shared<Translate>(box2, Vec3(130, 0, 65));
    world.add(box2);

    Camera cam;

    cam.aspect_ratio = 1.0;
    cam.image_width = 600;
    cam.samples_per_pixel = 200;
    cam.max_ray_bounces = 50;
    cam.background = Color(0,0,0);

    cam.vfov = 40;
    cam.lookfrom = Point3(278, 278, -800);
    cam.lookat = Point3(278, 278, 0);
    cam.vup = Vec3(0,1,0);

    cam.defocus_angle = 0;

    cam.render(world);
}

void cornell_smoke() {
    HittableList world;

    shared_ptr<Material> red = make_shared<Lambertian>(Color(0.65, 0.05, 0.05));
    shared_ptr<Material> white = make_shared<Lambertian>(Color(0.73, 0.73, 0.73));
    shared_ptr<Material> green = make_shared<Lambertian>(Color(0.12, 0.45, 0.15));
    shared_ptr<Material> light = make_shared<DiffuseLight>(Color(7, 7, 7));

    world.add(make_shared<Quad>(Point3(555,0,0), Vec3(0,555,0), Vec3(0,0,555), green));
    world.add(make_shared<Quad>(Point3(0,0,0), Vec3(0,555,0), Vec3(0,0,555), red));
    world.add(make_shared<Quad>(Point3(343, 554, 332), Vec3(-130,0,0), Vec3(0,0,-105), light));
    world.add(make_shared<Quad>(Point3(0,0,0), Vec3(555,0,0), Vec3(0,0,555), white));
    world.add(make_shared<Quad>(Point3(555,555,555), Vec3(-555,0,0), Vec3(0,0,-555), white));
    world.add(make_shared<Quad>(Point3(0,0,555), Vec3(555,0,0), Vec3(0,555,0), white));

    shared_ptr<Hittable> box1 = box(Point3(0, 0, 0), Point3(165, 330, 165), white);
    box1 = make_shared<RotateY>(box1, 15);
    box1 = make_shared<Translate>(box1, Vec3(265, 0, 295));

    shared_ptr<Hittable> box2 = box(Point3(0, 0, 0), Point3(165, 165, 165), white);
    box2 = make_shared<RotateY>(box2, -18);
    box2 = make_shared<Translate>(box2, Vec3(130, 0, 65));

    world.add(make_shared<ConstantMedium>(box1, 0.01, Color(0, 0, 0)));
    world.add(make_shared<ConstantMedium>(box2, 0.01, Color(1, 1, 1)));

    Camera cam;

    cam.aspect_ratio = 1.0;
    cam.image_width = 600;
    cam.samples_per_pixel = 100;
    cam.max_ray_bounces = 20;
    cam.background = Color(0,0,0);

    cam.vfov = 40;
    cam.lookfrom = Point3(278, 278, -800);
    cam.lookat = Point3(278, 278, 0);
    cam.vup = Vec3(0,1,0);

    cam.defocus_angle = 0;

    cam.render(world);
}

void final_chapt2_scene(int image_width, int samples_per_pixel, int max_depth) {
    HittableList boxes1;
    shared_ptr<Material> ground = make_shared<Lambertian>(Color(0.48, 0.83, 0.53));

    int boxes_per_side = 20;
    for (int i = 0; i < boxes_per_side; i++) {
        for (int j = 0; j < boxes_per_side; j++) {
            double w = 100.0;
            double x0 = -1000.0 + i * w;
            double z0 = -1000.0 + j * w;
            double y0 = 0.0;
            double x1 = x0 + w;
            double y1 = random_double(1, 101);
            double z1 = z0 + w;

            boxes1.add(box(Point3(x0, y0, z0), Point3(x1, y1, z1), ground));
        }
    }

    HittableList world;

    world.add(make_shared<BvhNode>(boxes1));

    shared_ptr<Material> light = make_shared<DiffuseLight>(Color(7, 7, 7));
    world.add(make_shared<Quad>(Point3(123, 554, 147), Vec3(300, 0, 0), Vec3(0, 0, 265), light));

    Point3 center1 = Point3(400, 400, 200);
    Point3 center2 = center1 + Vec3(30, 0, 0);
    shared_ptr<Material> sphere_mat = make_shared<Lambertian>(Color(0.7, 0.3, 0.1));
    world.add(make_shared<Sphere>(center1, center2, 50, sphere_mat));

    world.add(make_shared<Sphere>(Point3(260, 150, 45), 50, make_shared<Dielectric>(1.5)));
    world.add(make_shared<Sphere>(Point3(0, 150, 145), 50, make_shared<Metal>(Color(0.8, 0.8, 0.9), 1.0)));

    shared_ptr<Hittable> boundary = make_shared<Sphere>(Point3(360, 150, 145), 70, make_shared<Dielectric>(1.5));
    world.add(boundary);
    world.add(make_shared<ConstantMedium>(boundary, 0.2, Color(0.2, 0.4, 0.9)));
    boundary = make_shared<Sphere>(Point3(0, 0, 0), 5000, make_shared<Dielectric>(1.5));
    world.add(make_shared<ConstantMedium>(boundary, 0.0001, Color(1, 1, 1)));

    shared_ptr<Material> emat = make_shared<Lambertian>(make_shared<ImageTexture>("earthmap.jpg"));
    world.add(make_shared<Sphere>(Point3(400, 200, 400), 100, emat));
    shared_ptr<Texture> pertext = make_shared<NoiseTexture>(0.2);
    world.add(make_shared<Sphere>(Point3(220, 280, 300), 80, make_shared<Lambertian>(pertext)));

    HittableList boxes2;
    shared_ptr<Material> white = make_shared<Lambertian>(Color(0.73, 0.73, 0.73));
    int ns = 1000;
    for (int j = 0; j < ns; j++) {
        boxes2.add(make_shared<Sphere>(Point3::random(0, 165), 10, white));
    }

    world.add(make_shared<Translate>(make_shared<RotateY>(make_shared<BvhNode>(boxes2), 15), Vec3(-100, 270, 395)));

    Camera cam;

    cam.aspect_ratio = 1.0;
    cam.image_width = image_width;
    cam.samples_per_pixel = samples_per_pixel;
    cam.max_ray_bounces = max_depth;
    cam.background = Color(0,0,0);

    cam.vfov = 40;
    cam.lookfrom = Point3(478, 278, -600);
    cam.lookat = Point3(278, 278, 0);
    cam.vup = Vec3(0,1,0);

    cam.defocus_angle = 0;

    cam.render(world);
}

void render_blender_triangle_mesh() {
    HittableList world;
    HittableList mesh_objects;
    shared_ptr<Material> red_mat = std::make_shared<Lambertian>(Color(0.8, 0.2, 0.2));

    if (load_object_mesh("models/cube.obj", mesh_objects, red_mat)) {
        auto mesh_bvh = std::make_shared<BvhNode>(mesh_objects);
        world.add(mesh_bvh);
    }

    Camera cam;

    cam.aspect_ratio = 1.0;
    cam.image_width = 400;
    cam.samples_per_pixel = 100;
    cam.max_ray_bounces = 20;
    cam.background = Color(0.7, 0.8, 1.0);

    cam.vfov = 40;
    cam.lookfrom = Point3(3, 3, 5);
    cam.lookat = Point3(0, 1, 0);
    cam.vup = Vec3(0,1,0);

    cam.defocus_angle = 0;

    cam.render(world);
}

int main() {
    switch (13) {
        case 1: bouncing_spheres(); break;
        case 2: checkered_spheres(); break;
        case 3: moon(); break;
        case 4: perlin_spheres(); break;
        case 5: triangles(); break;
        case 6: hat(); break;
        case 7: poop_emoji(); break;
        case 8: quads(); break;
        case 9: simple_light(); break;
        case 10: cornell_box(); break;
        case 11: cornell_smoke(); break;
        case 12: final_chapt2_scene(800, 10000, 40); break;
        case 13: render_blender_triangle_mesh(); break;
    }
}