#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "common_constants.h"

class Sphere : public Hittable {
    private:
        Vec3 center;
        double radius;
    public:
        Sphere(const Vec3& center, double radius) : center(center), radius(std::max(0, radius)) {}

        bool hit(const Ray& r, double ray_tmin, double ray_tmax, HitRecord& rec) const override {
            Vec3 oc = center - r.origin();

            double a = r.direction().length_squared();
            double h = dot(r.direction(), oc);
            double c = oc.length_squared() - radius * radius;
            double discriminant = h * h - a * c;

            if (discriminant < 0) {
                return false;
            }

            double sqrtd = std::sqrt(discriminant);

            double root = (h - sqrtd) / a; // closer side of sphere
            if (root <= ray_tmin || ray_tmax <= root) {
                root = (h + sqrtd) / a; // farther side
                if (root <= ray_tmin || ray_tmax <= root) {
                    return false;
                }
            }

            rec.t = root;
            rec.p = r.at(rec.t);
            rec.normal = (rec.p - center) / radius;
            Vec3 outward_normal = (rec.p - center) / radius;
            rec.set_face_normal(r, outward_normal);

            return true;
        }
};

#endif