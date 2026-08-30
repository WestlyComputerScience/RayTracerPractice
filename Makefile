westly_ray_tracer: main.o
	g++ main.o -o westly_ray_tracer

main.o: main.cpp color.h common_constants.h hittable_list.h hittable.h ray.h sphere.h vec3.h interval.h camera.h material.h
	g++ -Wall -Werror -g main.cpp -c

clean: 
	rm -f *.o core westly_ray_tracer