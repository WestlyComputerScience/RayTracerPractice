CXXFLAGS = -Wall -Werror -Wno-unused-but-set-variable -g

# Switches all instances of reals to floats
ifdef USE_FLOAT
CXXFLAGS += -DUSE_FLOAT
endif

westly_ray_tracer: main.o
	g++ main.o -o westly_ray_tracer

main.o: aabb.h bvh.h camera.h color.h common_constants.h constant_medium.h hittable_list.h hittable.h interval.h main.cpp material.h perlin.h quad.h ray.h rtw_stb_image.h sphere.h texture.h triangle.h vec3.h
	g++ $(CXXFLAGS) main.cpp -c

clean: 
	rm -f *.o core westly_ray_tracer