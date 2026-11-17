#include "gpu_scene.cuh"

#include <stdexcept>
#include <string>

namespace {

void check_cuda_error(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " +
            cudaGetErrorString(error)
        );
    }
}

template <typename T>
void upload_array(
    T*& device_pointer,
    const std::vector<T>& host_data
) {
    if (host_data.empty()) {
        device_pointer = nullptr;
        return;
    }

    size_t bytes = host_data.size() * sizeof(T);

    check_cuda_error(
        cudaMalloc(
            reinterpret_cast<void**>(&device_pointer),
            bytes
        ),
        "cudaMalloc failed"
    );

    check_cuda_error(
        cudaMemcpy(
            device_pointer,
            host_data.data(),
            bytes,
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy to GPU failed"
    );
}

} // namespace

GPUDeviceScene upload_gpu_scene(const GPUScene& scene) {
    GPUDeviceScene device_scene{};

    device_scene.bvh_node_count = scene.bvh_nodes.size();
    device_scene.primitive_count = scene.primitives.size();
    device_scene.material_count = scene.materials.size();

    try {
        upload_array(
            device_scene.bvh_nodes,
            scene.bvh_nodes
        );

        upload_array(
            device_scene.primitives,
            scene.primitives
        );

        upload_array(
            device_scene.materials,
            scene.materials
        );
    }
    catch (...) {
        free_gpu_scene(device_scene);
        throw;
    }

    return device_scene;
}

void free_gpu_scene(GPUDeviceScene& scene) {
    if (scene.bvh_nodes != nullptr) {
        cudaFree(scene.bvh_nodes);
        scene.bvh_nodes = nullptr;
    }

    if (scene.primitives != nullptr) {
        cudaFree(scene.primitives);
        scene.primitives = nullptr;
    }

    if (scene.materials != nullptr) {
        cudaFree(scene.materials);
        scene.materials = nullptr;
    }

    scene.bvh_node_count = 0;
    scene.primitive_count = 0;
    scene.material_count = 0;
}
