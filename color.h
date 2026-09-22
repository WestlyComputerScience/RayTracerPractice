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
        Color(double r, double g, double b) : v(r, g, b) {}

        /**
        * Wraps an existing vector into a color object.
        */
        explicit Color(const Vec3& vec) : v(vec) {}

        // RGB getters
        double r() const { return v.x(); }
        double g() const { return v.y(); }
        double b() const { return v.z(); }

        // Standard vector operations.
        Color operator+(const Color& c) const { return Color(v + c.v); }
        Color operator*(double t) const { return Color(t * v); }
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
        static Color random(double min, double max) {
            return Color(Vec3::random(min, max));
        }
};

/**
* Scalar multiplication support.
*/
inline Color operator*(double t, const Color& c) {
    return c * t;
}


/**
* Converts a linear light component into gamma space.
*/
inline double linear_to_gamma(double linear_component) {
    if (linear_component > 0) {
        return std::sqrt(linear_component);
    }
    return 0;
}

/**
* Formats and writes a single pixel's RGB data to an output stream in PPM format.
*/
inline void write_color(std::ostream& out, const Color& pixel_color) {
    double r = pixel_color.r();
    double g = pixel_color.g();
    double b = pixel_color.b();

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