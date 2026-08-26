#ifndef COLOR_H
#define COLOR_H

#include "common_constants.h"

class Color {
    private:
        Vec3 v;

    public:
        Color(): v(0, 0, 0) {}
        Color(double r, double g, double b) : v(r, g, b) {}
        explicit Color(const Vec3& vec) : v(vec) {}

        double r() const { return v.x(); }
        double g() const { return v.y(); }
        double b() const { return v.z(); }

        Color operator+(const Color& c) const { return Color(v + c.v); }
        Color operator*(double t) const { return Color(t * v); }
        Color operator*(const Color& c) const { return Color(this->r() * c.r(), this->g() * c.g(), this->b() * c.b()); }
};

inline Color operator*(double t, const Color& c) {
    return c * t;
}

inline void write_color(std::ostream& out, const Color& pixel_color) {
    int rbyte = int(255.999 * pixel_color.r());
    int gbyte = int(255.999 * pixel_color.g());
    int bbyte = int(255.999 * pixel_color.b());

    out << rbyte << ' ' << gbyte << ' ' << bbyte << '\n';
}

#endif