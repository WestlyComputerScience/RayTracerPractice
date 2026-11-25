
#ifndef GPU_SCENE_BUILDER_H
#define GPU_SCENE_BUILDER_H

#include <memory>
#include <stdexcept>

#include "material.h"
#include "texture.h"
#include "gpu_scene.h"
#include "quad.h"
#include "sphere.h"
#include "hittable_list.h"
#include "gpu_bvh_builder.h"

inline Color get_solid_texture_color(
    const shared_ptr<Texture>& texture
) {
    auto solid = std::dynamic_pointer_cast<SolidColor>(texture);

    if (!solid) {
        throw std::runtime_error(
            "GPU renderer currently only supports SolidColor textures."
        );
    }

    return solid->get_albedo();
}

inline GPUMaterial convert_lambertian_to_gpu(
    const Lambertian& material
) {
    GPUMaterial gpu_material{};

    gpu_material.type = GPUMaterialType::Lambertian;

    const Color albedo = get_solid_texture_color(
        material.get_texture()
    );

    gpu_material.albedo_r = static_cast<float>(albedo.r());
    gpu_material.albedo_g = static_cast<float>(albedo.g());
    gpu_material.albedo_b = static_cast<float>(albedo.b());

    return gpu_material;
}

inline GPUMaterial convert_metal_to_gpu(
    const Metal& material
) {
    GPUMaterial gpu_material{};

    gpu_material.type = GPUMaterialType::Metal;

    const Color& albedo = material.get_albedo();

    gpu_material.albedo_r = static_cast<float>(albedo.r());
    gpu_material.albedo_g = static_cast<float>(albedo.g());
    gpu_material.albedo_b = static_cast<float>(albedo.b());

    gpu_material.roughness = static_cast<float>(material.get_fuzz());

    return gpu_material;
}

inline GPUMaterial convert_dielectric_to_gpu(
    const Dielectric& material
) {
    GPUMaterial gpu_material{};

    gpu_material.type = GPUMaterialType::Dielectric;

    gpu_material.refractive_index =
        static_cast<float>(material.get_refraction_index());

    return gpu_material;
}

inline GPUMaterial convert_diffuse_light_to_gpu(
    const DiffuseLight& material
) {
    GPUMaterial gpu_material{};

    gpu_material.type = GPUMaterialType::DiffuseLight;

    const Color emission = get_solid_texture_color(
        material.get_texture()
    );

    gpu_material.emission_r = static_cast<float>(emission.r());
    gpu_material.emission_g = static_cast<float>(emission.g());
    gpu_material.emission_b = static_cast<float>(emission.b());

    return gpu_material;
}

inline GPUMaterial convert_isotropic_to_gpu(
    const Isotropic& material
) {
    GPUMaterial gpu_material{};

    gpu_material.type = GPUMaterialType::Isotropic;

    const Color albedo = get_solid_texture_color(
        material.get_texture()
    );

    gpu_material.albedo_r = static_cast<float>(albedo.r());
    gpu_material.albedo_g = static_cast<float>(albedo.g());
    gpu_material.albedo_b = static_cast<float>(albedo.b());

    return gpu_material;
}

inline GPUMaterial convert_ggx_to_gpu(
    const GGXMaterial& material
) {
    GPUMaterial gpu_material{};

    gpu_material.type = GPUMaterialType::GGX;

    const Color albedo = get_solid_texture_color(
        material.get_albedo_texture()
    );

    gpu_material.albedo_r = static_cast<float>(albedo.r());
    gpu_material.albedo_g = static_cast<float>(albedo.g());
    gpu_material.albedo_b = static_cast<float>(albedo.b());

    gpu_material.roughness =
        static_cast<float>(material.get_roughness());

    gpu_material.metalness =
        static_cast<float>(material.get_metalness());

    return gpu_material;
}

inline int add_material_to_gpu_scene(
    GPUScene& scene,
    const shared_ptr<Material>& material
) {
    if (!material) {
        throw std::runtime_error(
            "Cannot register a null material."
        );
    }

    GPUMaterial gpu_material{};

    if (auto lambertian =
        std::dynamic_pointer_cast<Lambertian>(material)) {

        gpu_material = convert_lambertian_to_gpu(*lambertian);

    } else if (auto metal =
        std::dynamic_pointer_cast<Metal>(material)) {

        gpu_material = convert_metal_to_gpu(*metal);

    } else if (auto dielectric =
        std::dynamic_pointer_cast<Dielectric>(material)) {

        gpu_material = convert_dielectric_to_gpu(*dielectric);

    } else if (auto diffuse_light =
        std::dynamic_pointer_cast<DiffuseLight>(material)) {

        gpu_material = convert_diffuse_light_to_gpu(*diffuse_light);

    } else if (auto isotropic =
        std::dynamic_pointer_cast<Isotropic>(material)) {

        gpu_material = convert_isotropic_to_gpu(*isotropic);

    } else if (auto ggx =
        std::dynamic_pointer_cast<GGXMaterial>(material)) {

        gpu_material = convert_ggx_to_gpu(*ggx);

    } else {
        throw std::runtime_error(
            "Unsupported material type in GPU scene."
        );
    }

    int material_index = static_cast<int>(
        scene.materials.size()
    );

    scene.materials.push_back(gpu_material);

    return material_index;
}

// Converts a CPU Quad into GPU-compatible primitive data.
inline GPUPrimitive convert_quad_to_gpu(
    const Quad& quad,
    int material_index
) {
    GPUPrimitive primitive{};

    primitive.type = GPUPrimitiveType::Quad;

    Aabb bbox = quad.bounding_box();

    primitive.min_x = static_cast<float>(bbox.x.min);
    primitive.min_y = static_cast<float>(bbox.y.min);
    primitive.min_z = static_cast<float>(bbox.z.min);

    primitive.max_x = static_cast<float>(bbox.x.max);
    primitive.max_y = static_cast<float>(bbox.y.max);
    primitive.max_z = static_cast<float>(bbox.z.max);

    const Point3& Q = quad.get_Q();
    const Vec3& u = quad.get_u();
    const Vec3& v = quad.get_v();
    const Vec3& normal = quad.get_normal();

    // Q
    primitive.parameters[0] = static_cast<float>(Q.x());
    primitive.parameters[1] = static_cast<float>(Q.y());
    primitive.parameters[2] = static_cast<float>(Q.z());

    // u
    primitive.parameters[3] = static_cast<float>(u.x());
    primitive.parameters[4] = static_cast<float>(u.y());
    primitive.parameters[5] = static_cast<float>(u.z());

    // v
    primitive.parameters[6] = static_cast<float>(v.x());
    primitive.parameters[7] = static_cast<float>(v.y());
    primitive.parameters[8] = static_cast<float>(v.z());

    // Normal
    primitive.parameters[9] = static_cast<float>(normal.x());
    primitive.parameters[10] = static_cast<float>(normal.y());
    primitive.parameters[11] = static_cast<float>(normal.z());

    // Plane constant and area
    primitive.parameters[12] = static_cast<float>(quad.get_D());
    primitive.parameters[13] = static_cast<float>(quad.get_area());

    primitive.material_index = material_index;

    return primitive;
}

// Converts a CPU Sphere into GPU-compatible primitive data.
inline GPUPrimitive convert_sphere_to_gpu(
    const Sphere& sphere,
    int material_index
) {
    if (!sphere.is_stationary()) {
        throw std::runtime_error(
            "GPU scene builder currently only supports stationary spheres."
        );
    }

    GPUPrimitive primitive{};

    primitive.type = GPUPrimitiveType::Sphere;

    Aabb bbox = sphere.bounding_box();

    primitive.min_x = static_cast<float>(bbox.x.min);
    primitive.min_y = static_cast<float>(bbox.y.min);
    primitive.min_z = static_cast<float>(bbox.z.min);

    primitive.max_x = static_cast<float>(bbox.x.max);
    primitive.max_y = static_cast<float>(bbox.y.max);
    primitive.max_z = static_cast<float>(bbox.z.max);

    const Point3& center = sphere.get_center();

    primitive.parameters[0] = static_cast<float>(center.x());
    primitive.parameters[1] = static_cast<float>(center.y());
    primitive.parameters[2] = static_cast<float>(center.z());
    primitive.parameters[3] = static_cast<float>(sphere.get_radius());

    primitive.material_index = material_index;

    return primitive;
}

inline GPUPrimitive add_sphere_to_gpu_scene(
    GPUScene& scene,
    const Sphere& sphere
) {
    int material_index = add_material_to_gpu_scene(
        scene,
        sphere.get_material()
    );

    return convert_sphere_to_gpu(
        sphere,
        material_index
    );
}

inline GPUPrimitive add_quad_to_gpu_scene(
    GPUScene& scene,
    const Quad& quad
) {
    int material_index = add_material_to_gpu_scene(
        scene,
        quad.get_material()
    );

    return convert_quad_to_gpu(
        quad,
        material_index
    );
}

inline GPUScene build_gpu_scene_from_list(
    const HittableList& world
) {
    GPUScene scene{};

    for (const auto& object : world.objects) {
        GPUPrimitive primitive{};

        if (auto quad = std::dynamic_pointer_cast<Quad>(object)) {

            primitive = add_quad_to_gpu_scene(
                scene,
                *quad
            );

        } else if (auto sphere =
            std::dynamic_pointer_cast<Sphere>(object)) {

            primitive = add_sphere_to_gpu_scene(
                scene,
                *sphere
            );

        } else {
            throw std::runtime_error(
                "GPU scene builder currently only supports "
                "Quad and Sphere objects."
            );
        }

        scene.primitives.push_back(primitive);
    }

    build_gpu_bvh(scene);

    // Rebuild light indices after BVH construction because building the BVH
    // may reorder scene.primitives. Light indices must refer to the final
    // primitive array used by traversal and PDF sampling.
    scene.light_indices.clear();

    for (size_t i = 0; i < scene.primitives.size(); ++i) {
        const GPUPrimitive& primitive = scene.primitives[i];

        if (
            primitive.material_index >= 0 &&
            static_cast<size_t>(primitive.material_index) <
                scene.materials.size() &&
            scene.materials[primitive.material_index].type ==
                GPUMaterialType::DiffuseLight
        ) {
            scene.light_indices.push_back(static_cast<int>(i));
        }
    }

    return scene;
}
#endif