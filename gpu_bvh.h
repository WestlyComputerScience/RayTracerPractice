
#ifndef GPU_SCENE_H
#define GPU_SCENE_H

#include <cstdint>
#include <vector>

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

    primitive.parameters[0] = static_cast<float>(Q.x());
    primitive.parameters[1] = static_cast<float>(Q.y());
    primitive.parameters[2] = static_cast<float>(Q.z());

    primitive.parameters[3] = static_cast<float>(u.x());
    primitive.parameters[4] = static_cast<float>(u.y());
    primitive.parameters[5] = static_cast<float>(u.z());

    primitive.parameters[6] = static_cast<float>(v.x());
    primitive.parameters[7] = static_cast<float>(v.y());
    primitive.parameters[8] = static_cast<float>(v.z());

    primitive.parameters[9] = static_cast<float>(normal.x());
    primitive.parameters[10] = static_cast<float>(normal.y());
    primitive.parameters[11] = static_cast<float>(normal.z());

    primitive.parameters[12] = static_cast<float>(quad.get_D());
    primitive.parameters[13] = static_cast<float>(quad.get_area());

    primitive.material_index = material_index;

    return primitive;
}

// Identifies the type of primitive stored in the GPU scene.
enum class GPUPrimitiveType : uint32_t {
    Quad,
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
// The meaning of parameters depends on the primitive type.
struct GPUPrimitive {
    GPUPrimitiveType type;

    // Axis-aligned bounding box.
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;

    // Quad data:
    // parameters[0..2]  = Q
    // parameters[3..5]  = u
    // parameters[6..8]  = v
    // parameters[9..11] = normal
    // parameters[12]    = D
    // parameters[13]    = area

    // Other primitive types will define their own layouts.
    float parameters[16];

    int material_index;
};

// GPU material types.
enum class GPUMaterialType : uint32_t {
    Lambertian,
    Metal,
    Dielectric,
    DiffuseLight
};

// GPU-compatible material data.
struct GPUMaterial {
    GPUMaterialType type;

    float albedo_r;
    float albedo_g;
    float albedo_b;

    float roughness;
    float refractive_index;

    float emission_r;
    float emission_g;
    float emission_b;
};

// Complete GPU scene data.
struct GPUScene {
    std::vector<GPUBvhNode> bvh_nodes;
    std::vector<GPUPrimitive> primitives;
    std::vector<GPUMaterial> materials;
};

#endif
