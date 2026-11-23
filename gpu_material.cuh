
#ifndef GPU_MATERIAL_CUH
#define GPU_MATERIAL_CUH

#include "gpu_scene.h"
#include "gpu_intersection.cuh"
#include "gpu_traversal.cuh"
#include "gpu_vec3.cuh"
#include "gpu_random.cuh"
#include "gpu_sampling.cuh"
#include "gpu_pdf.cuh"

// Identifies the material PDF stored in a scatter record.
enum class GPUScatterPdfType {
    None,
    Cosine,
    Sphere,
    GGX
};

// GPU equivalent of the CPU ScatterRecord.
struct GPUScatterRecord {
    GPUVec3 attenuation;

    GPUScatterPdfType pdf_type;

    GPUCosinePdf cosine_pdf;
    GPUSpherePdf sphere_pdf;
    GPUGGXPdf ggx_pdf;

    bool skip_pdf;
    GPURay skip_pdf_ray;
};

// Constructs a GPU ray.
__device__ inline GPURay gpu_make_ray(
    const GPUVec3& origin,
    const GPUVec3& direction
) {
    return GPURay{
        origin.x,
        origin.y,
        origin.z,
        direction.x,
        direction.y,
        direction.z
    };
}

// Returns the incoming ray direction as a GPUVec3.
__device__ inline GPUVec3 gpu_ray_direction(
    const GPURay& ray
) {
    return GPUVec3{
        ray.direction_x,
        ray.direction_y,
        ray.direction_z
    };
}

// Returns the ray origin as a GPUVec3.
__device__ inline GPUVec3 gpu_ray_origin(
    const GPURay& ray
) {
    return GPUVec3{
        ray.origin_x,
        ray.origin_y,
        ray.origin_z
    };
}

// Returns the hit position as a GPUVec3.
__device__ inline GPUVec3 gpu_hit_position(
    const GPUTraversalResult& hit
) {
    return GPUVec3{
        hit.position_x,
        hit.position_y,
        hit.position_z
    };
}

// Returns the outward surface normal as a GPUVec3.
__device__ inline GPUVec3 gpu_hit_outward_normal(
    const GPUTraversalResult& hit
) {
    return GPUVec3{
        hit.normal_x,
        hit.normal_y,
        hit.normal_z
    };
}

// Matches the CPU HitRecord::set_face_normal() behavior.
// The returned normal always opposes the incoming ray.
__device__ inline GPUVec3 gpu_orient_normal(
    const GPURay& ray,
    const GPUTraversalResult& hit,
    bool& front_face
) {
    GPUVec3 outward_normal = gpu_hit_outward_normal(hit);
    GPUVec3 incoming_direction = gpu_ray_direction(ray);

    front_face =
        gpu_dot(incoming_direction, outward_normal) < 0.0f;

    return front_face
        ? outward_normal
        : -outward_normal;
}

// Schlick approximation used by the CPU Dielectric material.
__device__ inline float gpu_dielectric_reflectance(
    float cosine,
    float refraction_ratio
) {
    float r0 =
        (1.0f - refraction_ratio) /
        (1.0f + refraction_ratio);

    r0 *= r0;

    float one_minus_cosine = 1.0f - cosine;

    return r0 +
        (1.0f - r0) *
        one_minus_cosine *
        one_minus_cosine *
        one_minus_cosine *
        one_minus_cosine *
        one_minus_cosine;
}

// Constructs the GGX PDF basis, matching the CPU GGXPdf constructor.
__device__ inline GPUGGXPdf gpu_make_ggx_pdf(
    const GPUVec3& normal,
    const GPUVec3& view,
    float roughness
) {
    GPUVec3 a =
        fabsf(normal.x) > 0.9f
            ? GPUVec3{0.0f, 1.0f, 0.0f}
            : GPUVec3{1.0f, 0.0f, 0.0f};

    GPUVec3 tangent = gpu_unit_vector(
        gpu_cross(a, normal)
    );

    GPUVec3 bitangent = gpu_cross(
        normal,
        tangent
    );

    GPUGGXPdf pdf{};

    pdf.normal_x = normal.x;
    pdf.normal_y = normal.y;
    pdf.normal_z = normal.z;

    pdf.view_x = view.x;
    pdf.view_y = view.y;
    pdf.view_z = view.z;

    pdf.tangent_x = tangent.x;
    pdf.tangent_y = tangent.y;
    pdf.tangent_z = tangent.z;

    pdf.bitangent_x = bitangent.x;
    pdf.bitangent_y = bitangent.y;
    pdf.bitangent_z = bitangent.z;

    pdf.roughness = roughness;

    return pdf;
}

// Evaluates the material's emitted radiance.
// DiffuseLight emits only from its front-facing side.
__device__ inline GPUVec3 gpu_material_emitted(
    const GPUMaterial& material,
    bool front_face
) {
    if (
        material.type != GPUMaterialType::DiffuseLight ||
        !front_face
    ) {
        return GPUVec3{0.0f, 0.0f, 0.0f};
    }

    return GPUVec3{
        material.emission_r,
        material.emission_g,
        material.emission_b
    };
}

// Samples the material and fills its scatter record.
//
// Returns false when the material does not scatter or when
// the material's scattering conditions are not satisfied.
__device__ inline bool gpu_material_scatter(
    const GPUMaterial& material,
    const GPURay& incoming_ray,
    const GPUTraversalResult& hit,
    unsigned int& seed,
    GPUScatterRecord& scatter
) {
    scatter = GPUScatterRecord{};

    bool front_face = false;

    GPUVec3 normal = gpu_orient_normal(
        incoming_ray,
        hit,
        front_face
    );

    GPUVec3 incoming_direction = gpu_ray_direction(
        incoming_ray
    );

    GPUVec3 hit_position = gpu_hit_position(hit);

    GPUVec3 unit_direction = gpu_unit_vector(
        incoming_direction
    );

    // Lambertian material.
    if (material.type == GPUMaterialType::Lambertian) {
        scatter.attenuation = GPUVec3{
            material.albedo_r,
            material.albedo_g,
            material.albedo_b
        };

        scatter.pdf_type = GPUScatterPdfType::Cosine;

        scatter.cosine_pdf = GPUCosinePdf{
            normal.x,
            normal.y,
            normal.z
        };

        scatter.skip_pdf = false;

        return true;
    }

    // Metal material.
    if (material.type == GPUMaterialType::Metal) {
        GPUVec3 reflected = gpu_reflect(
            incoming_direction,
            normal
        );

        reflected = gpu_unit_vector(reflected);

        GPUVec3 fuzz_direction = gpu_random_unit_vector(
            seed
        );

        GPUVec3 scattered_direction =
            reflected +
            material.roughness * fuzz_direction;

        scatter.attenuation = GPUVec3{
            material.albedo_r,
            material.albedo_g,
            material.albedo_b
        };

        scatter.skip_pdf = true;

        scatter.skip_pdf_ray = gpu_make_ray(
            hit_position,
            scattered_direction
        );

        return true;
    }

    // Dielectric material.
    if (material.type == GPUMaterialType::Dielectric) {
        scatter.attenuation = GPUVec3{
            1.0f,
            1.0f,
            1.0f
        };

        scatter.skip_pdf = true;

        float refraction_ratio = front_face
            ? 1.0f / material.refractive_index
            : material.refractive_index;

        float cos_theta = fminf(
            gpu_dot(-unit_direction, normal),
            1.0f
        );

        float sin_theta = sqrtf(
            fmaxf(0.0f, 1.0f - cos_theta * cos_theta)
        );

        bool cannot_refract =
            refraction_ratio * sin_theta > 1.0f;

        GPUVec3 scattered_direction;

        if (
            cannot_refract ||
            gpu_dielectric_reflectance(
                cos_theta,
                refraction_ratio
            ) > gpu_random_float(seed)
        ) {
            scattered_direction = gpu_reflect(
                unit_direction,
                normal
            );
        } else {
            scattered_direction = gpu_refract(
                unit_direction,
                normal,
                refraction_ratio
            );
        }

        scatter.skip_pdf_ray = gpu_make_ray(
            hit_position,
            scattered_direction
        );

        return true;
    }

    // DiffuseLight does not scatter rays.
    if (material.type == GPUMaterialType::DiffuseLight) {
        return false;
    }

    // Isotropic material.
    if (material.type == GPUMaterialType::Isotropic) {
        scatter.attenuation = GPUVec3{
            material.albedo_r,
            material.albedo_g,
            material.albedo_b
        };

        scatter.pdf_type = GPUScatterPdfType::Sphere;
        scatter.sphere_pdf = GPUSpherePdf{};

        scatter.skip_pdf = false;

        return true;
    }

    // GGX material.
    if (material.type == GPUMaterialType::GGX) {
        GPUVec3 view = -unit_direction;

        float n_dot_v = gpu_dot(normal, view);

        if (n_dot_v <= 0.0f) {
            return false;
        }

        GPUVec3 albedo{
            material.albedo_r,
            material.albedo_g,
            material.albedo_b
        };

        // F0 = 0.04 * (1 - metalness) + albedo * metalness.
        GPUVec3 f0 =
            GPUVec3{0.04f, 0.04f, 0.04f} *
            (1.0f - material.metalness) +
            albedo * material.metalness;

        scatter.attenuation = f0;

        scatter.pdf_type = GPUScatterPdfType::GGX;

        scatter.ggx_pdf = gpu_make_ggx_pdf(
            normal,
            view,
            material.roughness
        );

        scatter.skip_pdf = false;

        return true;
    }

    return false;
}

// Generates a direction from the material's PDF.
__device__ inline bool gpu_material_pdf_generate(
    const GPUScatterRecord& scatter,
    unsigned int& seed,
    float& direction_x,
    float& direction_y,
    float& direction_z
) {
    if (scatter.pdf_type == GPUScatterPdfType::Cosine) {
        gpu_cosine_pdf_generate(
            scatter.cosine_pdf,
            seed,
            direction_x,
            direction_y,
            direction_z
        );

        return true;
    }

    if (scatter.pdf_type == GPUScatterPdfType::Sphere) {
        gpu_sphere_pdf_generate(
            seed,
            direction_x,
            direction_y,
            direction_z
        );

        return true;
    }

    if (scatter.pdf_type == GPUScatterPdfType::GGX) {
        gpu_ggx_pdf_generate(
            scatter.ggx_pdf,
            seed,
            direction_x,
            direction_y,
            direction_z
        );

        return true;
    }

    direction_x = 0.0f;
    direction_y = 0.0f;
    direction_z = 0.0f;

    return false;
}

// Evaluates the material's sampling PDF.
__device__ inline float gpu_material_pdf_value(
    const GPUScatterRecord& scatter,
    float direction_x,
    float direction_y,
    float direction_z
) {
    if (scatter.pdf_type == GPUScatterPdfType::Cosine) {
        return gpu_cosine_pdf_value(
            scatter.cosine_pdf,
            direction_x,
            direction_y,
            direction_z
        );
    }

    if (scatter.pdf_type == GPUScatterPdfType::Sphere) {
        return gpu_sphere_pdf_value();
    }

    if (scatter.pdf_type == GPUScatterPdfType::GGX) {
        return gpu_ggx_pdf_value(
            scatter.ggx_pdf,
            direction_x,
            direction_y,
            direction_z
        );
    }

    return 0.0f;
}

__device__ inline float gpu_mixture_pdf_value(
    const GPUMixturePdf& mixture,
    const GPUPrimitive* primitives,
    const int* light_indices,
    int light_count,
    const GPUHittablePdf& hittable_pdf,
    const GPUScatterRecord& material_scatter,
    float direction_x,
    float direction_y,
    float direction_z
) {
    float hittable_value = gpu_hittable_pdf_value(
        primitives,
        light_indices,
        light_count,
        hittable_pdf,
        direction_x,
        direction_y,
        direction_z
    );

    float material_value = gpu_material_pdf_value(
        material_scatter,
        direction_x,
        direction_y,
        direction_z
    );

    return 0.5f * hittable_value + 0.5f * material_value;
}

__device__ inline bool gpu_mixture_pdf_generate(
    GPUMixturePdf& mixture,
    const GPUPrimitive* primitives,
    const int* light_indices,
    int light_count,
    const GPUHittablePdf& hittable_pdf,
    const GPUScatterRecord& material_scatter,
    unsigned int& seed,
    float& direction_x,
    float& direction_y,
    float& direction_z
) {
    mixture.sample_hittable = gpu_random_float(seed) < 0.5f;

    if (mixture.sample_hittable) {
        return gpu_hittable_pdf_generate(
            primitives,
            light_indices,
            light_count,
            hittable_pdf,
            seed,
            direction_x,
            direction_y,
            direction_z
        );
    }

    return gpu_material_pdf_generate(
        material_scatter,
        seed,
        direction_x,
        direction_y,
        direction_z
    );
}

// Evaluates the material's scattering PDF.
// This corresponds to Material::scattering_pdf() on the CPU.
__device__ inline float gpu_material_scattering_pdf(
    const GPUMaterial& material,
    const GPUVec3& normal,
    const GPUVec3& scattered_direction
) {
    if (material.type == GPUMaterialType::Lambertian) {
        GPUVec3 direction = gpu_unit_vector(
            scattered_direction
        );

        float cosine = gpu_dot(normal, direction);

        return fmaxf(
            0.0f,
            cosine / 3.14159265358979323846f
        );
    }

    if (material.type == GPUMaterialType::Isotropic) {
        return gpu_sphere_pdf_value();
    }

    return 0.0f;
}

// Evaluates the material's scattering contribution.
//
// For ordinary materials this is attenuation * scattering_pdf.
// GGX uses the Cook-Torrance BRDF.
__device__ inline GPUVec3 gpu_material_scattering_value(
    const GPUMaterial& material,
    const GPUScatterRecord& scatter,
    const GPURay& incoming_ray,
    const GPUTraversalResult& hit,
    const GPUVec3& scattered_direction
) {
    GPUVec3 normal = gpu_hit_outward_normal(hit);

    bool front_face = false;

    normal = gpu_orient_normal(
        incoming_ray,
        hit,
        front_face
    );

    if (material.type == GPUMaterialType::GGX) {
        GPUVec3 view = gpu_unit_vector(
            -gpu_ray_direction(incoming_ray)
        );

        GPUVec3 light = gpu_unit_vector(
            scattered_direction
        );

        float n_dot_v = gpu_dot(normal, view);
        float n_dot_l = gpu_dot(normal, light);

        if (n_dot_v <= 1e-8f || n_dot_l <= 0.0f) {
            return GPUVec3{0.0f, 0.0f, 0.0f};
        }

        GPUVec3 half_vector = gpu_unit_vector(
            view + light
        );

        float v_dot_h = fmaxf(
            gpu_dot(view, half_vector),
            0.0f
        );

        float fresnel_factor = powf(
            1.0f - v_dot_h,
            5.0f
        );

        // Schlick Fresnel approximation.
        GPUVec3 f0 = scatter.attenuation;

        GPUVec3 fresnel =
            f0 +
            (GPUVec3{1.0f, 1.0f, 1.0f} - f0) *
            fresnel_factor;

        float D = gpu_ggx_distribution(
            scatter.ggx_pdf,
            half_vector.x,
            half_vector.y,
            half_vector.z
        );

        float G_view = gpu_ggx_geometry_smith_g1(
            scatter.ggx_pdf,
            n_dot_v
        );

        float G_light = gpu_ggx_geometry_smith_g1(
            scatter.ggx_pdf,
            n_dot_l
        );

        float G = G_view * G_light;

        float denominator = 4.0f * n_dot_v;

        return fresnel * (D * G / denominator);
    }

    float scattering_pdf = gpu_material_scattering_pdf(
        material,
        normal,
        scattered_direction
    );

    return scatter.attenuation * scattering_pdf;
}

#endif