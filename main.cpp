#include "common_constants.h"
#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"

int main() {
    HittableList world;

    world.add(make_shared<Sphere>(Vec3(0, 0, -1), 0.5));
    world.add(make_shared<Sphere>(Vec3(0, -100.5, -1), 100));

    Camera cam;

    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = 400;

    cam.render(world);
}