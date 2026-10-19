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

            // try nearest root
            real root = (-1 * b - sqrtd) / (2.0 * a);
            if (ray_t.surrounds(root)) {
                Point3 p = r.at(root);
                if (clipping_check(p, root)) {
                    rec.t = root;
                    rec.p = p;
                    Vec3 outward_normal = compute_normal(p);
                    rec.set_face_normal(r, outward_normal);
                    rec.mat = mat;
                    return true;
                }
            }

            // try second root
            root = (-1 * b + sqrtd) / (2.0 * a);
            if (ray_t.surrounds(root)) {
                Point3 p = r.at(root);
                if (clipping_check(p, root)) {
                    rec.t = root;
                    rec.p = p;
                    Vec3 outward_normal = compute_normal(p);
                    rec.set_face_normal(r, outward_normal);
                    rec.mat = mat;
                    return true;
                }
            }

            return false;
        }

        Aabb bounding_box() const override { return bbox; }
};

class Cone : public Quadric {
    private:
        real radius;
        real height;
        real k; // ratio of radius to height

        bool solve_coefficients(const Ray& r, real& a, real& b, real& c) const override {
            real ox = r.origin().x();
            real oy = r.origin().y();
            real oz = r.origin().z();

            real dx = r.direction().x();
            real dy = r.direction().y();
            real dz = r.direction().z();

            a = dx * dx + dz * dz - (k * k) * (dy * dy);
            b = 2.0 * (ox * dx + oz * dz - (k * k) * (oy * dy));
            c = ox * ox + oz * oz - (k * k) * (oy * oy);

            return true;
        }

        bool clipping_check(const Point3& p, real t) const override {
            // check if hit point falls within height bounds of cone
            return (p.y() >= 0.0 && p.y() <= height);
        }

        Vec3 compute_normal(const Point3& p) const override {
            real r_val = std::sqrt(p.x() * p.x() + p.z() * p.z());
            if (r_val == 0.0) return Vec3(0, 1, 0); // at apex

            real nx = p.x();
            real ny = -1 * k * r_val;
            real nz = p.z();

            return unit_vector(Vec3(nx, ny, nz));
        }
    public:
        Cone(real radius, real height, shared_ptr<Material> mat) : Quadric(mat), radius(radius), height(height), k(radius / height) {}

        Aabb bounding_box() const override {
            real r = radius;
            Point3 min_box(-1 * r, 0.0, -1 * r);
            Point3 max_box(r, height, r);
            return Aabb(min_box, max_box);
        }
};

#endif