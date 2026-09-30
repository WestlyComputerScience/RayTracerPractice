#ifndef XYZ_H
#define XYZ_H

#include "common_constants.h"

class XYZ {
    public:
        real x, y, z;

        Color toColor() const {
            real r =  3.2406 * x - 1.5372 * y - 0.4986 * z;
            real g = -0.9689 * x + 1.8758 * y + 0.0415 * z;
            real b =  0.0557 * x - 0.2040 * y + 1.0570 * z;

            return Color(r, g, b);
        }
};

#endif