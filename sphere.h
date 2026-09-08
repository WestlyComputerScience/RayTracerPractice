#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "common_constants.h"

class Sphere : public Hittable {
    private:
        Ray center;
        double radius;
        shared_ptr<Material> mat;
        Aabb bbox; // Bounding Box
    public:
        Sphere(const Point3& static_center, double radius, std::shared_ptr<Material> mat) 
        : center(static_center, Vec3(0, 0, 0)), radius(std::fmax(0, radius)), mat(mat) {
            Vec3 rvec = Vec3(radius, radius, radius);
            bbox = Aabb(static_center - rvec, static_center + rvec);
        }
        
        Sphere(const Point3& center1, const Point3& center2, double radius, std::shared_ptr<Material> mat) 
        : center(center1, center2 - center1), radius(std::fmax(0, radius)), mat(mat) {
            Vec3 rvec = Vec3(radius, radius, radius);
            Aabb box1(center.at(0) - rvec, center.at(0) + rvec);
            Aabb box2(center.at(1) - rvec, center.at(1) + rvec);
            bbox = Aabb(box1, box2);
        }

        bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const override {
            Point3 current_center = center.at(r.time());
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
            Vec3 outward_normal = (rec.p - current_center) / radius;
            rec.set_face_normal(r, outward_normal);
            get_sphere_uv(outward_normal, rec.u, rec.v);
            rec.mat = mat;

            return true;
        }

        Aabb bounding_box() const override { return bbox; }

        static void get_sphere_uv(const Point3& p, double& u, double& v) {
            double theta = std::acos(-p.y());
            double phi = std::atan2(-p.z(), p.x()) + pi;

            u = phi / (2 * pi);
            v = theta / pi;
        }
};

#endif