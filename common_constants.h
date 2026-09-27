#ifndef COMMON_CONSTANTS_H
#define COMMON_CONSTANTS_H

/**
* A utility header used to aggregate common imports and methods.
*/

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

using std::make_shared;
using std::shared_ptr;

#ifdef USE_FLOAT
    using real = float;
#else
    using real = double;
#endif

const real infinity = std::numeric_limits<real>::infinity();

const real pi = 3.1415926535897932385;

inline real degrees_to_radians(real degrees) {
    return degrees * pi / 180.0;
}

inline real random_real() {
    static std::uniform_real_distribution<real> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

inline real random_real(real min, real max) {
    return min + (max - min) * random_real();
}

inline int random_int(int min, int max) {
    return int(random_real(min, max + 1));
}

#include "interval.h"
#include "vec3.h"
#include "color.h"
#include "ray.h"

#endif