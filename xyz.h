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

            r = std::clamp(r, static_cast<real>(0.0), r);
            g = std::clamp(g, static_cast<real>(0.0), g);
            b = std::clamp(b, static_cast<real>(0.0), b);

            return Color(r, g, b);
        }
};

#endif