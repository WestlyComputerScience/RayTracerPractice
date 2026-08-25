westly_ray_tracer: main.o
	g++ main.o -o westly_ray_tracer

main.o: main.cpp color.h vec3.h
	g++ -Wall -Werror -g main.cpp -c