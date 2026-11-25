
#ifndef GPU_SCENE_H
#define GPU_SCENE_H

#include <cstdint>
#include <vector>

// Identifies the type of primitive stored in the GPU scene.
enum class GPUPrimitiveType : uint32_t {
    Quad,
    Sphere,
    Cone,
    Cylinder,
    Hyperboloid,
    HyperbolicParaboloid,
    Torus,
    BohemianDome,
    ChairSurface,
    KummerSurface,
    RomanSurface,
    GoursatSurface,
    TangleCube
};

// GPU-compatible BVH node.
struct GPUBvhNode {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;

    int left_child;
    int right_child;

    int primitive_start;
    int primitive_count;
};

// GPU-compatible primitive data.
struct GPUPrimitive {
    GPUPrimitiveType type;

    // Axis-aligned bounding box.
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;

    // The meaning of these parameters depends on the primitive type.
    float parameters[16];

    int material_index;
};

// GPU material types.
enum class GPUMaterialType : uint32_t {
    Lambertian,
    Metal,
    Dielectric,
    DiffuseLight,
    Isotropic,
    GGX
};

// GPU-compatible material data.
struct GPUMaterial {
    GPUMaterialType type;

    float albedo_r;
    float albedo_g;
    float albedo_b;

    // Metal fuzz or GGX roughness, depending on material type.
    float roughness;

    // Dielectric index of refraction.
    float refractive_index;

    // GGX metallic parameter.
    float metalness;

    float emission_r;
    float emission_g;
    float emission_b;
};

// Complete GPU scene data.
struct GPUScene {
    std::vector<GPUBvhNode> bvh_nodes;
    std::vector<GPUPrimitive> primitives;
    std::vector<GPUMaterial> materials;
    std::vector<int> light_indices;
};

#endif