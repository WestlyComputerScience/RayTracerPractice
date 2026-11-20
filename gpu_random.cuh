#ifndef GPU_RANDOM_CUH
#define GPU_RANDOM_CUH

__device__ inline unsigned int gpu_random_uint(unsigned int& seed) {
    // Xorshift32 requires a nonzero state.
    if (seed == 0u) {
        seed = 0x6D2B79F5u;
    }

    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;

    return seed;
}

__device__ inline float gpu_random_float(unsigned int& seed) {
    return static_cast<float>(
        gpu_random_uint(seed) >> 8
    ) * (1.0f / 16777216.0f);
}

#endif