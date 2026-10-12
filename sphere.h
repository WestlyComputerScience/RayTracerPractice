#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "common_constants.h"

/**
* Implementation of the sphere object.
*/
class Sphere : public Hittable {
    private:
        Ray center;
        real radius;
        shared_ptr<Material> mat;
        Aabb bbox; // Bounding Box

        static Vec3 random_to_sphere(real radius, real distance_squared) {
            real r1 = random_real();
            real r2 = random_real();
            real z = 1 + r2 * (std::sqrt(1 - radius * radius / distance_squared) - 1);

            real phi = 2 * pi * r1;
            real x = std::cos(phi) * std::sqrt(1 - z * z);
            real y = std::sin(phi) * std::sqrt(1 - z * z);

            return Vec3(x, y, z);
        }

    public:
        /**
        * Builds a stationary sphere.
        */
        Sphere(const Point3& static_center, real radius, std::shared_ptr<Material> mat) 
        : center(static_center, Vec3(0, 0, 0)), radius(std::fmax(0, radius)), mat(mat) {
            Vec3 rvec = Vec3(radius, radius, radius);
            bbox = Aabb(static_center - rvec, static_center + rvec);
        }
        
        /**
        * Builds a moving sphere traveling between 2 points.
        */
        Sphere(const Point3& center1, const Point3& center2, real radius, std::shared_ptr<Material> mat) 
        : center(center1, center2 - center1), radius(std::fmax(0, radius)), mat(mat) {
            Vec3 rvec = Vec3(radius, radius, radius);
            Aabb box1(center.at(0) - rvec, center.at(0) + rvec);
            Aabb box2(center.at(1) - rvec, center.at(1) + rvec);
            bbox = Aabb(box1, box2);
        }

        /**
        * Calculates an intersection between a ray and a sphere.
        */
        bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const override {
            Point3 current_center = center.at(r.time()); // Accounts for motion blur
            Vec3 oc = current_center - r.origin();

            // evaluates the discriminant
            real a = r.direction().length_squared();
            real h = dot(r.direction(), oc);
            real c = oc.length_squared() - radius * radius;
            real discriminant = h * h - a * c;

            if (discriminant < 0) {
                return false;
            }

            real sqrtd = std::sqrt(discriminant);

            real root = (h - sqrtd) / a; // closer side of sphere
            if (!ray_t.surrounds(root)) {
                root = (h + sqrtd) / a; // farther side
                if (!ray_t.surrounds(root)) {
                    return false;
                }
            }

            // populate the hit record
            rec.t = root;
            rec.p = r.at(rec.t);
            Vec3 outward_normal = (rec.p - current_center) / radius;
            rec.set_face_normal(r, outward_normal);
            get_sphere_uv(outward_normal, rec.u, rec.v);
            rec.mat = mat;

            return true;
        }

        Aabb bounding_box() const override { return bbox; }

        /**
        * Converts a uniot sphere surface vector into a 2D UV texture
        */
        static void get_sphere_uv(const Point3& p, real& u, real& v) {
            real theta = std::acos(-p.y());
            real phi = std::atan2(-p.z(), p.x()) + pi;

            u = phi / (2 * pi);
            v = theta / pi;
        }

        real pdf_value(const Point3& origin, const Vec3& direction) const override {
            // TODO: this only works for stationary spheres

            HitRecord rec;
            if (!this->hit(Ray(origin, direction), Interval(0.001, infinity), rec)) return 0;

            real dist_squared = (center.at(0) - origin).length_squared();
            real cos_theta_max = std::sqrt(1 - radius * radius / dist_squared);
            real solid_angle = 2 * pi * (1 - cos_theta_max);

            return 1 / solid_angle;
        }

        Vec3 random(const Point3& origin) const override {
            Vec3 direction = center.at(0) - origin;
            real distance_squared = direction.length_squared();
            Onb uvw(direction);
            return uvw.transform(random_to_sphere(radius, distance_squared));
        }
};

#endif