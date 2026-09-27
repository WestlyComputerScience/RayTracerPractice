#ifndef VEC3_H
#define VEC3_H

#include "common_constants.h"

#include <array>

/**
* The heart of the ray tracer, this is a standard vector class.
*/
class Vec3 {
    public:
        real e[3];

        Vec3(): e{0, 0, 0} {}
        Vec3(real e0, real e1, real e2) : e{e0, e1, e2} {}

        real x() const { return e[0]; }
        real y() const { return e[1]; }
        real z() const { return e[2]; }

        Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
        real operator[](int i) const { return e[i]; }
        real& operator[](int i) { return e[i]; }

        Vec3& operator+=(const Vec3& v) {
            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];
            return *this;
        }

        Vec3& operator*=(real t) {
            e[0] *= t;
            e[1] *= t;
            e[2] *= t;
            return *this;
        }

        Vec3& operator/=(real t) {
            return *this *= 1/t;
        }

        real length() const {
            return std::sqrt(length_squared());
        }

        real length_squared() const {
            return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
        }

        bool near_zero() const {
            real s = 1e-8;
            return (std::fabs(e[0]) < s) && (std::fabs(e[1]) < s) && (std::fabs(e[2]) < s);
        }

        static Vec3 random() {
            return Vec3(random_real(), random_real(), random_real());
        }

        static Vec3 random(real min, real max) {
            return Vec3(random_real(min, max), random_real(min, max), random_real(min, max));
        }
};

using Point3 = Vec3;

inline std::ostream& operator<<(std::ostream& out, const Vec3& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

inline Vec3 operator+(const Vec3& u, const Vec3& v) {
    return Vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

inline Vec3 operator-(const Vec3& u, const Vec3& v) {
    return Vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

inline Vec3 operator*(const Vec3& u, const Vec3& v) {
    return Vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

inline Vec3 operator*(const Vec3& v, const real t) {
    return Vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

inline Vec3 operator*(real t, const Vec3& v) {
    return v * t;
}

inline Vec3 operator/(const Vec3& v, const real t) {
    return (1.0/t) * v;
}

inline real dot(const Vec3& u, const Vec3& v) {
    return  u.e[0] * v.e[0] + 
            u.e[1] * v.e[1] + 
            u.e[2] * v.e[2];
}

inline Vec3 cross(const Vec3& u, const Vec3& v) {
    return Vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

inline Vec3 unit_vector(const Vec3& v) {
    return v / v.length();
}

inline Vec3 random_unit_vector() {
    while (true) {
        Vec3 p = Vec3::random(-1 ,1); // randomized point -1 to 1 on all axes
        real lensq = p.length_squared();
        if (1e-160 < lensq && lensq <= 1) {
            return p / sqrt(lensq); // normalizes vector to length 1
        }
    }
}

inline Vec3 random_on_hemisphere(const Vec3& normal) {
    Vec3 on_unit_sphere = random_unit_vector();
    if (dot(on_unit_sphere, normal) > 0.0) { // in the same hemisphere in normal
        return on_unit_sphere;
    } else {
        return -on_unit_sphere;
    }
}

inline Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - 2 * dot(v, n) * n;
}

inline Vec3 refract(const Vec3& uv, const Vec3& n, real etai_over_etat) {
    real cost_theta = std::fmin(dot(-uv, n), 1.0);
    Vec3 r_out_perp = etai_over_etat * (uv + cost_theta * n);
    Vec3 r_out_parallel = -std::sqrt(std::fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

inline Vec3 random_in_unit_disk() {
    while (true) {
        Vec3 p = Vec3(random_real(-1, 1), random_real(-1, 1), 0);
        if (p.length_squared() < 1) {
            return p;
        }
    }
}

/**
* Computes a * b - c * d, with high numerical accuracy.
*/
inline real difference_of_products(const real& a, const real& b, const real& c, const real& d) {
    real cd = c * d;
    real err = std::fma(c, d, -cd);
    real dop = std::fma(a, b, -cd);
    return dop + err;
}

/**
* An array containing a permutation of indicies 0, 1, 2 maps index 0 to y, index 1 to z, and index 2 to x.
* Used to compute the dominant coordinate axis and permute the triangles vertices so the say points primarily down +Z axis.
*/
inline Vec3 permute(const Vec3& v, std::array<int, 3> perm) {
    return Vec3(v[perm[0]], v[perm[1]], v[perm[2]]);
}

inline Vec3 abs(const Vec3& v) {
    return Vec3(std::abs(v.x()), std::abs(v.y()), std::abs(v.z()));
}

inline int max_component_index(const Vec3& v) {
    if (v.x() > v.y()) {
        return (v.x() > v.z()) ? 0 : 2;
    } else {
        return (v.y() > v.z()) ? 1 : 2;
    }
}

#endif