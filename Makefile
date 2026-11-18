
CXX = g++
NVCC = nvcc

CXXFLAGS = -Wall -Werror -Wno-unused-but-set-variable -fopenmp -g
NVCCFLAGS = -std=c++17 -arch=sm_120 -g

# Switches all instances of reals to floats
ifdef USE_FLOAT
    CXXFLAGS += -DUSE_FLOAT
    NVCCFLAGS += -DUSE_FLOAT
endif

# Use 20 nm spectral sampling instead of 1 nm
ifdef SPECTRAL_20NM
    CXXFLAGS += -DSPECTRAL_20NM
    NVCCFLAGS += -DSPECTRAL_20NM
endif

# Existing CPU renderer
westly_ray_tracer: main.o
	$(CXX) $(CXXFLAGS) main.o -o westly_ray_tracer

main.o: aabb.h bvh.h camera.h color.h common_constants.h constant_medium.h cube_map.h gpu_bvh_builder.h gpu_scene_builder.h gpu_scene.h hittable_list.h hittable.h interval.h main.cpp material.h onb.h pdf.h perlin.h quad.h quadrics.h quartic.h ray.h rtw_stb_image.h spectrum.h sphere.h texture.h triangle.h vec3.h
	$(CXX) $(CXXFLAGS) main.cpp -c

# CUDA test executable
cuda_test: cuda_test.cu
	$(NVCC) $(NVCCFLAGS) cuda_test.cu -o cuda_test

# CUDA GPU scene memory module
gpu_scene.o: gpu_scene.cu gpu_scene.cuh gpu_scene.h
	$(NVCC) $(NVCCFLAGS) -c gpu_scene.cu -o gpu_scene.o

gpu_upload_test.o: gpu_upload_test.cu gpu_scene.cuh gpu_scene_builder.h gpu_intersection.cuh
	$(NVCC) $(NVCCFLAGS) -c gpu_upload_test.cu -o gpu_upload_test.o

gpu_upload_test: gpu_upload_test.o gpu_scene.o
	$(NVCC) $(NVCCFLAGS) gpu_upload_test.o gpu_scene.o -o gpu_upload_test

gpu_intersection_test: gpu_intersection_test.cu
	$(NVCC) $(NVCCFLAGS) gpu_intersection_test.cu -o gpu_intersection_test

clean:
	rm -f *.o core westly_ray_tracer cuda_test
