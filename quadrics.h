#ifndef QUADRIC_H
#define QUADRIC_H

#include "common_constants.h"

/**
* Abstract class for all quadrics.
*/
class Quadric : public Hittable {
    private:
        shared_ptr<Material> mat;
        Aabb bbox;
    protected:
        /**
        * Computes quadric coefficients
        */
        virtual bool solve_coefficients(const Ray& r, real& a, real& b, real& c) const = 0;

        /**
        * Checks for clipping/bounding, returns true of the hit point at param t is valid.
        */
        virtual bool clipping_check(const Point3& p, real t) const { return true; }

        /**
        * Computes surface normal at a given point on the quadric.
        */
        virtual Vec3 compute_normal(const Point3& p) const = 0;
    public:
        Quadric(shared_ptr<Material> mat) : mat(mat)  {}

        bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const override {
            real a, b, c;
            if (!solve_coefficients(r, a, b, c)) return false;

            real discriminant = b * b - 4.0 * a * c;
            if (discriminant < 0.0) return false;

            real sqrtd = std::sqrt(discriminant);

            // find nearest root
            real root = (-1 * b - sqrtd) / (2.0 * a);
            if (!ray_t.surrounds(root)) {
                root = (-1 * b + sqrtd) / (2.0 * a);
                if (!ray_t.surrounds(root)) return false;
            }

            Point3 p = r.at(root);
            if (!clipping_check(p, root)) {
                // try second root
                if (root == (-1 * b - sqrtd) / (2.0 * a)) {
                    root = (-1 * B + sqrtd) / (2.0 * a);
                    if (ray_t.surrounds(root)) {
                        p = r.at(root);
                        if (clipping_check(p, root)) {
                            rec.t = root;
                            rec.p = p;
                            Vec3 outward_normal = compute_normal(p);
                            rec.set_face_normal(r, outward_normal);
                            rec.mat = mat;
                            return true;
                        }
                    }
                }
            }

            return false;
        }

        Aabb bounding_box() const override { return bbox; }
};

#endif