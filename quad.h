#ifndef QUAD_H
#define QUAD_H

#include "common_constants.h"

class Quad : public Hittable {
    private:
        Point3 Q;
        Vec3 u, v;
        Vec3 w;
        shared_ptr<Material> mat;
        Aabb bbox;
        Vec3 normal;
        double D;
    public:
        Quad(const Point3& Q, const Vec3& u, const Vec3& v, shared_ptr<Material> mat) : Q(Q), u(u), v(v), mat(mat) {
            Vec3 n = cross(u, v);
            normal = unit_vector(n);
            D = dot(normal, Q);
            w = n / dot(n, n);

            set_bounding_box();
        }

        virtual void set_bounding_box() {
            Aabb bbox_diagonal1 = Aabb(Q, Q + u + v);
            Aabb bbox_diagonal2 = Aabb(Q + u, Q + v);
            bbox = Aabb(bbox_diagonal1, bbox_diagonal2);
        }

        Aabb bounding_box() const override { return bbox; }

        bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const override {
            double denom = dot(normal, r.direction());

            if (std::fabs(denom) < 1e-8) return false; // no hit if ray is parallel to the plane

            double t = (D - dot(normal, r.origin())) / denom;
            if (!ray_t.contains(t)) return false; // if the hit point param t is outside ray interval

            // determine if the hit lies within the planear shape using it's coords
            Point3 intersection = r.at(t);
            Vec3 planar_hitpt_vector = intersection - Q;
            double alpha = dot(w, cross(planar_hitpt_vector, v));
            double beta = dot(w, cross(u, planar_hitpt_vector));

            if (!is_interior(alpha, beta, rec)) return false;

            // Ray hits 2D shape; set the rest of the hit record and return true
            rec.t = t;
            rec.p = intersection;
            rec.mat = mat;
            rec.set_face_normal(r, normal);

            return true;
        }

        virtual bool is_interior(double a, double b, HitRecord& rec) const {
            Interval unit_interval = Interval(0, 1);

            // Given the hit point in plane coords, return false if it's outside the primitive.
            // Otherwise, set HitRecord UV coords and return true;
            if (!unit_interval.contains(a) || !unit_interval.contains(b)) return false;

            rec.u = a;
            rec.v = b;
            return true;
        }
};

#endif