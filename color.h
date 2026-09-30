#ifndef COLOR_H
#define COLOR_H

#include "common_constants.h"

/**
* Dedicated RGB wrapper built around a 3D vector.
*/
class Color {
    private:
        Vec3 v;

    public:
        /**
        * By default, a color is black.
        */
        Color(): v(0, 0, 0) {}

        /**
        * Create a color given RGB.
        */
        Color(real r, real g, real b) : v(r, g, b) {}

        /**
        * Wraps an existing vector into a color object.
        */
        explicit Color(const Vec3& vec) : v(vec) {}

        // RGB getters
        real r() const { return v.x(); }
        real g() const { return v.y(); }
        real b() const { return v.z(); }

        // Standard vector operations.
        Color operator+(const Color& c) const { return Color(v + c.v); }
        Color operator*(real t) const { return Color(t * v); }
        Color operator*(const Color& c) const { return Color(this->r() * c.r(), this->g() * c.g(), this->b() * c.b()); }
        Color& operator+=(const Color& e) {
            v[0] += e.v[0];
            v[1] += e.v[1];
            v[2] += e.v[2];
            return *this;
        }

        /**
        * Outputs a random color.
        */
        static Color random() {
            return Color(Vec3::random());
        }

        /**
        * Outputs a color within a picked range.
        */
        static Color random(real min, real max) {
            return Color(Vec3::random(min, max));
        }
};

/**
* Scalar multiplication support.
*/
inline Color operator*(real t, const Color& c) {
    return c * t;
}


/**
* Converts a linear light component into gamma space. TODO: update description
*/
inline real linear_to_gamma(real linear_component) {
    if (linear_component <= 0.0031308) {
        return 12.92 * linear_component;
    }

    return 1.055 * std::pow(linear_component, 1.0 / 2.4) - 0.055;
}

/**
* Formats and writes a single pixel's RGB data to an output stream in PPM format.
*/
inline void write_color(std::ostream& out, const Color& pixel_color) {
    real r = pixel_color.r();
    real g = pixel_color.g();
    real b = pixel_color.b();

    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    static const Interval intensity(0.000, 0.999);
    int rbyte = int(256 * intensity.clamp(r));
    int gbyte = int(256 * intensity.clamp(g));
    int bbyte = int(256 * intensity.clamp(b));

    out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

#endif