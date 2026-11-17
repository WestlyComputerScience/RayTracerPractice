
#include <iostream>

#include "common_constants.h"
#include "vec3.h"
#include "interval.h"
#include "ray.h"
#include "aabb.h"

#include "gpu_scene_builder.h"
#include "gpu_scene.cuh"

int main() {
    // Create a small CPU scene containing two quads.
    HittableList world;

    shared_ptr<Material> red =
        make_shared<Lambertian>(Color(1.0, 0.0, 0.0));

    shared_ptr<Material> green =
        make_shared<Lambertian>(Color(0.0, 1.0, 0.0));

    world.add(make_shared<Quad>(
        Point3(0, 0, 0),
        Vec3(2, 0, 0),
        Vec3(0, 2, 0),
        red
    ));

    world.add(make_shared<Quad>(
        Point3(3, 0, 0),
        Vec3(2, 0, 0),
        Vec3(0, 2, 0),
        green
    ));

    // Convert the CPU scene into GPU-compatible records.
    GPUScene cpu_scene = build_gpu_scene_from_list(world);

    std::cout << "CPU scene:\n";
    std::cout << "BVH nodes: " << cpu_scene.bvh_nodes.size() << '\n';
    std::cout << "Primitives: " << cpu_scene.primitives.size() << '\n';
    std::cout << "Materials: " << cpu_scene.materials.size() << '\n';

    // Upload the scene to CUDA device memory.
    GPUDeviceScene device_scene = upload_gpu_scene(cpu_scene);

    std::cout << "\nGPU scene:\n";
    std::cout << "BVH nodes: " << device_scene.bvh_node_count << '\n';
    std::cout << "Primitives: " << device_scene.primitive_count << '\n';
    std::cout << "Materials: " << device_scene.material_count << '\n';

    // Verify the allocations.
    bool valid =
        device_scene.bvh_nodes != nullptr &&
        device_scene.primitives != nullptr &&
        device_scene.materials != nullptr;

    std::cout << "\nGPU allocations: "
              << (valid ? "SUCCESS" : "FAILED") << '\n';

    // Release GPU memory.
    free_gpu_scene(device_scene);

    std::cout << "GPU memory released.\n";

    return valid ? 0 : 1;
}
