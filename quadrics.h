#ifndef QUADRIC_H
#define QUADRIC_H

#include "common_constants.h"

/**
* Abstract class for all quadrics.
*/
class Quadric : public Hittable {
    private:
        shared_ptr<Material> mat;
    protected:
        Aabb bbox;
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
        Point3 center;
        real radius;
        real height;
        real k; // ratio of radius to height

        bool solve_coefficients(const Ray& r, real& a, real& b, real& c) const override {
            Point3 local_origin = r.origin() - center;

            real ox = local_origin.x();
            real oy = local_origin.y();
            real oz = local_origin.z();

            real dx = r.direction().x();
            real dy = r.direction().y();
            real dz = r.direction().z();

            a = dx * dx + dz * dz - (k * k) * (dy * dy);
            b = 2.0 * (ox * dx + oz * dz - (k * k) * (oy * dy));
            c = ox * ox + oz * oz - (k * k) * (oy * oy);

            return true;
        }

        bool clipping_check(const Point3& p, real t) const override {
            real local_y = p.y() - center.y();

            // check if hit point falls within height bounds of cone
            return (local_y >= 0.0 && local_y <= height);
        }

        Vec3 compute_normal(const Point3& p) const override {
            Point3 local_p = p - center;
            real r_val = std::sqrt(local_p.x() * local_p.x() + local_p.z() * local_p.z());
            if (r_val == 0.0) return Vec3(0, 1, 0); // at apex

            real nx = local_p.x();
            real ny = -1 * k * r_val;
            real nz = local_p.z();

            return unit_vector(Vec3(nx, ny, nz));
        }
    public:
        /**
         * Constructor for a cone centered at (0, 0, 0).
         */
        Cone(real radius, real height, shared_ptr<Material> mat) : Quadric(mat), center(Point3(0, 0, 0)), radius(radius), height(height), k(radius / height) {
            Vec3 r_vec(radius, radius, radius);
            bbox = Aabb(center - r_vec, center + Vec3(radius, height, radius));
        }

        /**
         * Constructor for a cone centered at a specified point.
         */
        Cone(const Point3& location, real radius, real height, std::shared_ptr<Material> mat) : Quadric(mat), center(location), radius(radius), height(height), k(radius / height) {
            Vec3 r_vec(radius, radius, radius);
            Point3 min_point = center - r_vec;
            Point3 max_point = center + Vec3(radius, height, radius);
            bbox = Aabb(min_point, max_point);
        }

        Aabb bounding_box() const override { return bbox; }
};

class Cylinder : public Quadric {
    private:
        Point3 center;
        real radius;
        real height;

        bool solve_coefficients(const Ray& r, real &a, real &b, real &c) const override {
            Point3 local_origin = r.origin() - center;

            real ox = local_origin.x();
            real oz = local_origin.z();

            real dx = r.direction().x();
            real dz = r.direction().z();

            a = dx * dx + dz * dz;
            b = 2.0 * (ox * dx + oz * dz);
            c = ox * ox + oz * oz - radius * radius;

            return true;
        }

        bool clipping_check(const Point3& p, real t) const override {
            real local_y = p.y() - center.y();

            return (local_y >= 0.0 && local_y <= height);
        }

        Vec3 compute_normal(const Point3& p) const override {
            Point3 local_p = p - center;

            return unit_vector(Vec3(local_p.x(), 0.0, local_p.z()));
        }

    public:
        /**
         * Constructor for a cylinder centered at (0, 0, 0).
         */
        Cylinder(real radius, real height, shared_ptr<Material> mat) : Quadric(mat), center(Point3(0, 0, 0)), radius(radius), height(height) {
            Point3 min_point = center - Vec3(radius, 0.0, radius);
            Point3 max_point = center + Vec3(radius, height, radius);
            bbox = Aabb(min_point, max_point);
        }

        /**
         * Constructor for a cylinder centered at a specified point location.
         */
        Cylinder(const Point3& location, real radius, real height, shared_ptr<Material> mat) : Quadric(mat), center(location), radius(radius), height(height) {
            Point3 min_point = center - Vec3(radius, 0.0, radius);
            Point3 max_point = center + Vec3(radius, height, radius);
            bbox = Aabb(min_point, max_point);
        }

        Aabb bounding_box() const override { return bbox; }
};

#endif