#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "common_constants.h"

class Sphere : public Hittable {
    private:
        Ray center;
        double radius;
        shared_ptr<Material> mat;
    public:
        Sphere(const Vec3& static_center, double radius, std::shared_ptr<Material> mat) 
        : center(static_center, Vec3(0, 0, 0)), radius(std::max(0.0, radius)), mat(mat) { }
        
        Sphere(const Vec3& center1, const Vec3& center2, double radius, std::shared_ptr<Material> mat) 
        : center(center1, center2 - center1), radius(std::max(0.0, radius)), mat(mat) { }

        bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const override {
            Vec3 current_center = center.at(r.time());
            Vec3 oc = current_center - r.origin();

            double a = r.direction().length_squared();
            double h = dot(r.direction(), oc);
            double c = oc.length_squared() - radius * radius;
            double discriminant = h * h - a * c;

            if (discriminant < 0) {
                return false;
            }

            double sqrtd = std::sqrt(discriminant);

            double root = (h - sqrtd) / a; // closer side of sphere
            if (!ray_t.surrounds(root)) {
                root = (h + sqrtd) / a; // farther side
                if (!ray_t.surrounds(root)) {
                    return false;
                }
            }

            rec.t = root;
            rec.p = r.at(rec.t);
            rec.normal = (rec.p - current_center) / radius;
            rec.set_face_normal(r, rec.normal);
            rec.mat = mat;

            return true;
        }
};

#endif