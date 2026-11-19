#ifndef GPU_SCENE_CUH
#define GPU_SCENE_CUH

#include <cstddef>

#include <cuda_runtime.h>

#include "gpu_scene.h"

// Holds GPU memory pointers for an uploaded scene.
struct GPUDeviceScene {
    GPUBvhNode* bvh_nodes = nullptr;
    GPUPrimitive* primitives = nullptr;
    GPUMaterial* materials = nullptr;
    int* light_indices = nullptr;

    size_t bvh_node_count = 0;
    size_t primitive_count = 0;
    size_t material_count = 0;
    size_t light_count = 0;
};

// Uploads a CPU GPUScene to CUDA device memory.
GPUDeviceScene upload_gpu_scene(const GPUScene& scene);

// Releases all CUDA allocations belonging to a GPUDeviceScene.
void free_gpu_scene(GPUDeviceScene& scene);

#endif