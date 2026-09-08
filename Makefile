westly_ray_tracer: main.o
	g++ main.o -o westly_ray_tracer

main.o: aabb.h bvh.h camera.h color.h common_constants.h hittable_list.h hittable.h interval.h main.cpp material.h ray.h rtw_stb_image.h sphere.h texture.h vec3.h
	g++ -Wall -Werror -Wno-unused-but-set-variable -g main.cpp -c

clean: 
	rm -f *.o core westly_ray_tracer