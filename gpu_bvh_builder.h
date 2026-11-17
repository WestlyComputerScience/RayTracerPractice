
#ifndef GPU_BVH_BUILDER_H
#define GPU_BVH_BUILDER_H

#include <algorithm>
#include <cstddef>
#include <limits>

#include "gpu_scene.h"

// Returns the minimum bounding-box coordinate along an axis.
inline float gpu_box_min(
    const GPUPrimitive& primitive,
    int axis
) {
    if (axis == 1) return primitive.min_y;
    if (axis == 2) return primitive.min_z;
    return primitive.min_x;
}

// Returns the maximum bounding-box coordinate along an axis.
inline float gpu_box_max(
    const GPUPrimitive& primitive,
    int axis
) {
    if (axis == 1) return primitive.max_y;
    if (axis == 2) return primitive.max_z;
    return primitive.max_x;
}

// Builds one node and returns its index in the flattened array.
inline int build_gpu_bvh_node(
    GPUScene& scene,
    size_t start,
    size_t end
) {
    int node_index = static_cast<int>(scene.bvh_nodes.size());

    // Reserve this node before recursively building its children.
    scene.bvh_nodes.emplace_back();

    GPUBvhNode node{};

    node.min_x = std::numeric_limits<float>::infinity();
    node.min_y = std::numeric_limits<float>::infinity();
    node.min_z = std::numeric_limits<float>::infinity();

    node.max_x = -std::numeric_limits<float>::infinity();
    node.max_y = -std::numeric_limits<float>::infinity();
    node.max_z = -std::numeric_limits<float>::infinity();

    node.left_child = -1;
    node.right_child = -1;

    node.primitive_start = 0;
    node.primitive_count = 0;

    // Calculate the combined bounding box.
    for (size_t i = start; i < end; i++) {
        const GPUPrimitive& primitive = scene.primitives[i];

        node.min_x = std::min(node.min_x, primitive.min_x);
        node.min_y = std::min(node.min_y, primitive.min_y);
        node.min_z = std::min(node.min_z, primitive.min_z);

        node.max_x = std::max(node.max_x, primitive.max_x);
        node.max_y = std::max(node.max_y, primitive.max_y);
        node.max_z = std::max(node.max_z, primitive.max_z);
    }

    size_t primitive_count = end - start;

    // Leaves contain at most two primitives.
    if (primitive_count <= 2) {
        node.primitive_start = static_cast<int>(start);
        node.primitive_count = static_cast<int>(primitive_count);

        scene.bvh_nodes[node_index] = node;
        return node_index;
    }

    // Find the longest bounding-box axis.
    float size_x = node.max_x - node.min_x;
    float size_y = node.max_y - node.min_y;
    float size_z = node.max_z - node.min_z;

    int axis = 0;

    if (size_y > size_x && size_y > size_z) {
        axis = 1;
    } else if (size_z > size_x && size_z > size_y) {
        axis = 2;
    }

    // Sort primitives along the selected axis.
    std::sort(
        scene.primitives.begin() + start,
        scene.primitives.begin() + end,
        [axis](const GPUPrimitive& a, const GPUPrimitive& b) {
            return gpu_box_min(a, axis) < gpu_box_min(b, axis);
        }
    );

    // Split at the median.
    size_t mid = start + primitive_count / 2;

    node.left_child = build_gpu_bvh_node(scene, start, mid);
    node.right_child = build_gpu_bvh_node(scene, mid, end);

    scene.bvh_nodes[node_index] = node;

    return node_index;
}

// Builds the complete flattened GPU BVH.
inline void build_gpu_bvh(GPUScene& scene) {
    scene.bvh_nodes.clear();

    if (scene.primitives.empty()) {
        return;
    }

    build_gpu_bvh_node(scene, 0, scene.primitives.size());
}

#endif
