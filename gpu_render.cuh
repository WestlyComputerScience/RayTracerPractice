
#ifndef GPU_RENDER_CUH
#define GPU_RENDER_CUH

#include "gpu_camera.h"
#include "gpu_camera.cuh"
#include "gpu_traversal.cuh"
#include "gpu_material.cuh"

constexpr int GPU_MAX_PATH_DEPTH = 64;

// GPU-compatible pixel color.
struct GPUColor {
    float r;
    float g;
    float b;
};


__device__ inline GPUVec3 gpu_trace_path(
    const GPUCamera& camera,
    GPURay ray,
    int max_depth,
    const GPUBvhNode* nodes,
    const GPUPrimitive* primitives,
    const GPUMaterial* materials,
    const int* light_indices,
    int node_count,
    int primitive_count,
    int material_count,
    int light_count,
    unsigned int& seed
) {
    // Each stack entry represents one recursive CPU ray_color() call
    // waiting for its child call to return.
    GPUVec3 emission_stack[GPU_MAX_PATH_DEPTH];
    GPUVec3 scattering_factor_stack[GPU_MAX_PATH_DEPTH];

    int stack_size = 0;

    // The recursive CPU implementation returns black at depth zero.
    GPUVec3 result{0.0f, 0.0f, 0.0f};

    if (max_depth <= 0) {
        return result;
    }

    // Keep stack accesses within the allocated local arrays.
    int depth_limit = max_depth;

    if (depth_limit > GPU_MAX_PATH_DEPTH) {
        depth_limit = GPU_MAX_PATH_DEPTH;
    }

    for (int depth = 0; depth < depth_limit; depth++) {
        GPUTraversalResult hit = gpu_bvh_hit(
            nodes,
            primitives,
            node_count,
            primitive_count,
            ray,
            0.001f,
            1.0e30f
        );

        if (hit.stack_overflow) {
            return GPUVec3{1.0f, 0.0f, 1.0f};
        }

        // A ray that misses returns the background.
        if (!hit.hit) {
            result = GPUVec3{
                camera.background_r,
                camera.background_g,
                camera.background_b
            };

            break;
        }

        if (
            hit.primitive_index < 0 ||
            hit.primitive_index >= primitive_count
        ) {
            return GPUVec3{1.0f, 0.0f, 1.0f};
        }

        const GPUPrimitive& primitive =
            primitives[hit.primitive_index];

        int material_index = primitive.material_index;

        if (
            material_index < 0 ||
            material_index >= material_count
        ) {
            return GPUVec3{1.0f, 0.0f, 1.0f};
        }

        const GPUMaterial& material =
            materials[material_index];

        // Determine whether the ray hit the front or back face.
        bool front_face = false;

        gpu_orient_normal(
            ray,
            hit,
            front_face
        );

        // Calculate emission from the current surface.
        GPUVec3 emitted = gpu_material_emitted(
            material,
            front_face
        );

        GPUScatterRecord scatter{};

        bool did_scatter = gpu_material_scatter(
            material,
            ray,
            hit,
            seed,
            scatter
        );

        // Metal and dielectric use skip_pdf.
        // Their CPU implementation directly returns the emission
        // of the secondary hit, or the background if it misses.
        if (did_scatter && scatter.skip_pdf) {
            GPUTraversalResult secondary_hit = gpu_bvh_hit(
                nodes,
                primitives,
                node_count,
                primitive_count,
                scatter.skip_pdf_ray,
                0.001f,
                1.0e30f
            );

            if (secondary_hit.stack_overflow) {
                return GPUVec3{1.0f, 0.0f, 1.0f};
            }

            if (!secondary_hit.hit) {
                result = GPUVec3{
                    camera.background_r,
                    camera.background_g,
                    camera.background_b
                };
            }
            else {
                if (
                    secondary_hit.primitive_index < 0 ||
                    secondary_hit.primitive_index >= primitive_count
                ) {
                    return GPUVec3{1.0f, 0.0f, 1.0f};
                }

                const GPUPrimitive& secondary_primitive =
                    primitives[secondary_hit.primitive_index];

                int secondary_material_index =
                    secondary_primitive.material_index;

                if (
                    secondary_material_index < 0 ||
                    secondary_material_index >= material_count
                ) {
                    return GPUVec3{1.0f, 0.0f, 1.0f};
                }

                bool secondary_front_face = false;

                gpu_orient_normal(
                    scatter.skip_pdf_ray,
                    secondary_hit,
                    secondary_front_face
                );

                result = gpu_material_emitted(
                    materials[secondary_material_index],
                    secondary_front_face
                );
            }

            // Do not include this surface's emission or scattering.
            // The result will still be propagated through any parent
            // recursive calls below.
            break;
        }

        // A material that does not scatter returns its emission.
        if (!did_scatter) {
            result = emitted;
            break;
        }

        GPUVec3 hit_position = gpu_hit_position(hit);

        GPUHittablePdf hittable_pdf{
            hit_position.x,
            hit_position.y,
            hit_position.z
        };

        GPUMixturePdf mixture{};

        float direction_x = 0.0f;
        float direction_y = 0.0f;
        float direction_z = 0.0f;

        bool generated = gpu_mixture_pdf_generate(
            mixture,
            primitives,
            light_indices,
            light_count,
            hittable_pdf,
            scatter,
            seed,
            direction_x,
            direction_y,
            direction_z
        );

        // The CPU returns the current surface's emission when
        // PDF generation cannot produce a valid direction.
        if (!generated) {
            result = emitted;
            break;
        }

        float pdf_value = gpu_mixture_pdf_value(
            mixture,
            primitives,
            light_indices,
            light_count,
            hittable_pdf,
            scatter,
            direction_x,
            direction_y,
            direction_z
        );

        if (pdf_value <= 1e-8f) {
            result = emitted;
            break;
        }

        GPUVec3 scattered_direction{
            direction_x,
            direction_y,
            direction_z
        };

        GPUVec3 scattering_value =
            gpu_material_scattering_value(
                material,
                scatter,
                ray,
                hit,
                scattered_direction
            );

        GPUVec3 scattering_factor =
            scattering_value * (1.0f / pdf_value);

        // Save the current recursive call before tracing its child.
        emission_stack[stack_size] = emitted;

        scattering_factor_stack[stack_size] =
            scattering_factor;

        stack_size++;

        // The CPU's recursive call at depth zero returns black.
        // Therefore, no more rays need to be traced at this depth.
        if (depth == depth_limit - 1) {
            result = GPUVec3{0.0f, 0.0f, 0.0f};
            break;
        }

        // Continue tracing the child ray.
        ray = gpu_make_ray(
            hit_position,
            scattered_direction
        );
    }

    // Unwind the simulated recursion stack.
    //
    // Each parent computes:
    // emitted + scattering_factor * child_result
    //
    // This also correctly propagates the special skip_pdf result
    // through any parent recursive calls.
    for (int i = stack_size - 1; i >= 0; i--) {
        result =
            emission_stack[i] +
            scattering_factor_stack[i] * result;
    }

    return result;
}

// Full GPU path-tracing renderer.
// One CUDA thread handles one pixel.
__global__ void gpu_render_path_tracing(
    GPUCamera camera,
    const GPUBvhNode* nodes,
    const GPUPrimitive* primitives,
    const GPUMaterial* materials,
    const int* light_indices,
    int node_count,
    int primitive_count,
    int material_count,
    int light_count,
    GPUColor* pixels
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int pixel_count =
        camera.image_width * camera.image_height;

    if (index >= pixel_count) {
        return;
    }

    if (camera.sqrt_spp <= 0 || camera.max_ray_bounces < 0) {
        pixels[index] = GPUColor{1.0f, 0.0f, 1.0f};
        return;
    }

    int i = index % camera.image_width;
    int j = index / camera.image_width;

    int sample_count =
        camera.sqrt_spp * camera.sqrt_spp;

    GPUVec3 accumulated{0.0f, 0.0f, 0.0f};

    for (int s_j = 0; s_j < camera.sqrt_spp; s_j++) {
        for (int s_i = 0; s_i < camera.sqrt_spp; s_i++) {
            int sample_index =
                s_j * camera.sqrt_spp + s_i;

            unsigned int seed =
                static_cast<unsigned int>(
                    index * sample_count + sample_index + 1
                );

            float offset_x;
            float offset_y;

            gpu_sample_square_stratified(
                camera,
                s_i,
                s_j,
                seed,
                offset_x,
                offset_y
            );

            GPUVec3 pixel_position =
                camera.pixel00_loc +
                (i + offset_x) * camera.pixel_delta_u +
                (j + offset_y) * camera.pixel_delta_v;

            GPUVec3 ray_origin = camera.center;

            if (camera.defocus_angle > 0.0f) {
                ray_origin = gpu_defocus_disk_sample(
                    camera,
                    seed
                );
            }

            GPUVec3 ray_direction =
                pixel_position - ray_origin;

            GPURay ray = gpu_make_ray(
                ray_origin,
                ray_direction
            );

            GPUVec3 sample_color = gpu_trace_path(
                camera,
                ray,
                camera.max_ray_bounces,
                nodes,
                primitives,
                materials,
                light_indices,
                node_count,
                primitive_count,
                material_count,
                light_count,
                seed
            );

            accumulated = accumulated + sample_color;
        }
    }

    float reciprocal_samples =
        1.0f / static_cast<float>(sample_count);

    accumulated = accumulated * reciprocal_samples;

    pixels[index] = GPUColor{
        accumulated.x,
        accumulated.y,
        accumulated.z
    };
}

// Generates a camera ray and calculates its background color.
// One CUDA thread handles one pixel.
__global__ void gpu_render_background(
    GPUCamera camera,
    GPUColor* pixels
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int pixel_count = camera.image_width * camera.image_height;

    if (index >= pixel_count) {
        return;
    }

    pixels[index] = GPUColor{
        camera.background_r,
        camera.background_g,
        camera.background_b
    };
}

// Generates camera rays and renders a scene using flat material colors.
__global__ void gpu_render_scene(
    GPUCamera camera,
    const GPUBvhNode* nodes,
    const GPUPrimitive* primitives,
    const GPUMaterial* materials,
    int node_count,
    int primitive_count,
    int material_count,
    GPUColor* pixels
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int pixel_count = camera.image_width * camera.image_height;

    if (index >= pixel_count) {
        return;
    }

    int i = index % camera.image_width;
    int j = index / camera.image_width;

    float pixel_x =
        camera.pixel00_loc.x +
        i * camera.pixel_delta_u.x +
        j * camera.pixel_delta_v.x;

    float pixel_y =
        camera.pixel00_loc.y +
        i * camera.pixel_delta_u.y +
        j * camera.pixel_delta_v.y;

    float pixel_z =
        camera.pixel00_loc.z +
        i * camera.pixel_delta_u.z +
        j * camera.pixel_delta_v.z;

    GPURay ray{
        camera.center.x,
        camera.center.y,
        camera.center.z,
        pixel_x - camera.center.x,
        pixel_y - camera.center.y,
        pixel_z - camera.center.z
    };

    GPUTraversalResult hit = gpu_bvh_hit(
        nodes,
        primitives,
        node_count,
        primitive_count,
        ray,
        0.001f,
        1.0e30f
    );

    if (hit.stack_overflow) {
        pixels[index] = GPUColor{1.0f, 0.0f, 1.0f};
        return;
    }

    if (!hit.hit) {
        pixels[index] = GPUColor{
            camera.background_r,
            camera.background_g,
            camera.background_b
        };
        return;
    }

    if (hit.primitive_index < 0 ||
        hit.primitive_index >= primitive_count) {
        pixels[index] = GPUColor{1.0f, 0.0f, 1.0f};
        return;
    }

    const GPUPrimitive& primitive =
        primitives[hit.primitive_index];

    int material_index = primitive.material_index;

    if (material_index < 0 ||
        material_index >= material_count) {
        pixels[index] = GPUColor{1.0f, 0.0f, 1.0f};
        return;
    }

    const GPUMaterial& material = materials[material_index];

    if (material.type == GPUMaterialType::DiffuseLight) {
        pixels[index] = GPUColor{
            material.emission_r,
            material.emission_g,
            material.emission_b
        };
    }
    else {
        pixels[index] = GPUColor{
            material.albedo_r,
            material.albedo_g,
            material.albedo_b
        };
    }
}

// Renders a scene using multiple stratified camera samples per pixel.
// Supports thin-lens depth of field through aperture sampling.
// One CUDA thread handles one pixel.
__global__ void gpu_render_scene_multisample(
    GPUCamera camera,
    const GPUBvhNode* nodes,
    const GPUPrimitive* primitives,
    const GPUMaterial* materials,
    int node_count,
    int primitive_count,
    int material_count,
    GPUColor* pixels
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int pixel_count = camera.image_width * camera.image_height;

    if (index >= pixel_count) {
        return;
    }

    if (camera.sqrt_spp <= 0) {
        pixels[index] = GPUColor{1.0f, 0.0f, 1.0f};
        return;
    }

    int i = index % camera.image_width;
    int j = index / camera.image_width;

    int sample_count = camera.sqrt_spp * camera.sqrt_spp;

    float accumulated_r = 0.0f;
    float accumulated_g = 0.0f;
    float accumulated_b = 0.0f;

    for (int s_j = 0; s_j < camera.sqrt_spp; s_j++) {
        for (int s_i = 0; s_i < camera.sqrt_spp; s_i++) {

            int sample_index =
                s_j * camera.sqrt_spp + s_i;

            unsigned int seed =
                static_cast<unsigned int>(
                    index * sample_count + sample_index + 1
                );

            // Generate the stratified subpixel offset.
            float offset_x;
            float offset_y;

            gpu_sample_square_stratified(
                camera,
                s_i,
                s_j,
                seed,
                offset_x,
                offset_y
            );

            // Find the sampled position on the camera's image plane.
            float pixel_x =
                camera.pixel00_loc.x +
                (i + offset_x) * camera.pixel_delta_u.x +
                (j + offset_y) * camera.pixel_delta_v.x;

            float pixel_y =
                camera.pixel00_loc.y +
                (i + offset_x) * camera.pixel_delta_u.y +
                (j + offset_y) * camera.pixel_delta_v.y;

            float pixel_z =
                camera.pixel00_loc.z +
                (i + offset_x) * camera.pixel_delta_u.z +
                (j + offset_y) * camera.pixel_delta_v.z;

            // Sample the camera aperture when defocus is enabled.
            GPUVec3 ray_origin = camera.center;

            if (camera.defocus_angle > 0.0f) {
                ray_origin = gpu_defocus_disk_sample(camera, seed);
            }

            GPURay ray{
                ray_origin.x,
                ray_origin.y,
                ray_origin.z,
                pixel_x - ray_origin.x,
                pixel_y - ray_origin.y,
                pixel_z - ray_origin.z
            };

            // Trace this camera sample through the GPU BVH.
            GPUTraversalResult hit = gpu_bvh_hit(
                nodes,
                primitives,
                node_count,
                primitive_count,
                ray,
                0.001f,
                1.0e30f
            );

            if (hit.stack_overflow) {
                pixels[index] = GPUColor{1.0f, 0.0f, 1.0f};
                return;
            }

            if (!hit.hit) {
                accumulated_r += camera.background_r;
                accumulated_g += camera.background_g;
                accumulated_b += camera.background_b;
                continue;
            }

            if (hit.primitive_index < 0 ||
                hit.primitive_index >= primitive_count) {
                pixels[index] = GPUColor{1.0f, 0.0f, 1.0f};
                return;
            }

            const GPUPrimitive& primitive =
                primitives[hit.primitive_index];

            int material_index = primitive.material_index;

            if (material_index < 0 ||
                material_index >= material_count) {
                pixels[index] = GPUColor{1.0f, 0.0f, 1.0f};
                return;
            }

            const GPUMaterial& material =
                materials[material_index];

            if (material.type == GPUMaterialType::DiffuseLight) {
                accumulated_r += material.emission_r;
                accumulated_g += material.emission_g;
                accumulated_b += material.emission_b;
            }
            else {
                accumulated_r += material.albedo_r;
                accumulated_g += material.albedo_g;
                accumulated_b += material.albedo_b;
            }
        }
    }

    // Average all camera samples for this pixel.
    float reciprocal_samples = 1.0f / sample_count;

    pixels[index] = GPUColor{
        accumulated_r * reciprocal_samples,
        accumulated_g * reciprocal_samples,
        accumulated_b * reciprocal_samples
    };
}

// Renders surface normals as RGB colors.
// One CUDA thread handles one pixel.
__global__ void gpu_render_normals(
    GPUCamera camera,
    const GPUBvhNode* nodes,
    const GPUPrimitive* primitives,
    int node_count,
    int primitive_count,
    GPUColor* pixels
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int pixel_count = camera.image_width * camera.image_height;

    if (index >= pixel_count) {
        return;
    }

    int i = index % camera.image_width;
    int j = index / camera.image_width;

    float pixel_x =
        camera.pixel00_loc.x +
        i * camera.pixel_delta_u.x +
        j * camera.pixel_delta_v.x;

    float pixel_y =
        camera.pixel00_loc.y +
        i * camera.pixel_delta_u.y +
        j * camera.pixel_delta_v.y;

    float pixel_z =
        camera.pixel00_loc.z +
        i * camera.pixel_delta_u.z +
        j * camera.pixel_delta_v.z;

    GPURay ray{
        camera.center.x,
        camera.center.y,
        camera.center.z,
        pixel_x - camera.center.x,
        pixel_y - camera.center.y,
        pixel_z - camera.center.z
    };

    GPUTraversalResult hit = gpu_bvh_hit(
        nodes,
        primitives,
        node_count,
        primitive_count,
        ray,
        0.001f,
        1.0e30f
    );

    if (hit.stack_overflow) {
        pixels[index] = GPUColor{1.0f, 0.0f, 1.0f};
        return;
    }

    if (!hit.hit) {
        pixels[index] = GPUColor{
            camera.background_r,
            camera.background_g,
            camera.background_b
        };
        return;
    }

    pixels[index] = GPUColor{
        0.5f * (hit.normal_x + 1.0f),
        0.5f * (hit.normal_y + 1.0f),
        0.5f * (hit.normal_z + 1.0f)
    };
}

// Renders Lambertian materials using a directional light.
// Surface normals are oriented against the incoming camera ray.
__global__ void gpu_render_diffuse(
    GPUCamera camera,
    const GPUBvhNode* nodes,
    const GPUPrimitive* primitives,
    const GPUMaterial* materials,
    int node_count,
    int primitive_count,
    int material_count,
    GPUColor* pixels
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int pixel_count = camera.image_width * camera.image_height;

    if (index >= pixel_count) {
        return;
    }

    int i = index % camera.image_width;
    int j = index / camera.image_width;

    float pixel_x =
        camera.pixel00_loc.x +
        i * camera.pixel_delta_u.x +
        j * camera.pixel_delta_v.x;

    float pixel_y =
        camera.pixel00_loc.y +
        i * camera.pixel_delta_u.y +
        j * camera.pixel_delta_v.y;

    float pixel_z =
        camera.pixel00_loc.z +
        i * camera.pixel_delta_u.z +
        j * camera.pixel_delta_v.z;

    GPURay ray{
        camera.center.x,
        camera.center.y,
        camera.center.z,
        pixel_x - camera.center.x,
        pixel_y - camera.center.y,
        pixel_z - camera.center.z
    };

    GPUTraversalResult hit = gpu_bvh_hit(
        nodes,
        primitives,
        node_count,
        primitive_count,
        ray,
        0.001f,
        1.0e30f
    );

    if (hit.stack_overflow) {
        pixels[index] = GPUColor{1.0f, 0.0f, 1.0f};
        return;
    }

    if (!hit.hit) {
        pixels[index] = GPUColor{
            camera.background_r,
            camera.background_g,
            camera.background_b
        };
        return;
    }

    if (hit.primitive_index < 0 ||
        hit.primitive_index >= primitive_count) {
        pixels[index] = GPUColor{1.0f, 0.0f, 1.0f};
        return;
    }

    const GPUPrimitive& primitive =
        primitives[hit.primitive_index];

    int material_index = primitive.material_index;

    if (material_index < 0 ||
        material_index >= material_count) {
        pixels[index] = GPUColor{1.0f, 0.0f, 1.0f};
        return;
    }

    const GPUMaterial& material = materials[material_index];

    // Emissive surfaces return their emitted radiance.
    if (material.type == GPUMaterialType::DiffuseLight) {
        pixels[index] = GPUColor{
            material.emission_r,
            material.emission_g,
            material.emission_b
        };
        return;
    }

    // This diffuse implementation supports Lambertian materials.
    if (material.type != GPUMaterialType::Lambertian) {
        pixels[index] = GPUColor{1.0f, 0.0f, 1.0f};
        return;
    }

    // Retrieve the surface normal.
    float nx = hit.normal_x;
    float ny = hit.normal_y;
    float nz = hit.normal_z;

    // Orient the normal against the incoming ray.
    float normal_dot_ray =
        nx * ray.direction_x +
        ny * ray.direction_y +
        nz * ray.direction_z;

    if (normal_dot_ray > 0.0f) {
        nx = -nx;
        ny = -ny;
        nz = -nz;
    }

    // Direction from the surface toward the directional light.
    // This vector is normalized.
    constexpr float light_x = -0.40824829f;
    constexpr float light_y =  0.40824829f;
    constexpr float light_z = -0.81649658f;

    // Lambertian diffuse intensity.
    float diffuse_intensity =
        nx * light_x +
        ny * light_y +
        nz * light_z;

    // A surface facing away from the light receives no direct light.
    if (diffuse_intensity < 0.0f) {
        diffuse_intensity = 0.0f;
    }

    // Apply diffuse lighting to the material albedo.
    pixels[index] = GPUColor{
        material.albedo_r * diffuse_intensity,
        material.albedo_g * diffuse_intensity,
        material.albedo_b * diffuse_intensity
    };
}

#endif
