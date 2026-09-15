#ifndef QUAD_H
#define QUAD_H

#include "common_constants.h"
#include "hittable.h"
#include "hittable_list.h"

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

inline shared_ptr<HittableList> box(const Point3& a, const Point3& b, shared_ptr<Material> mat) {
    shared_ptr<HittableList> sides = make_shared<HittableList>();

    Point3 min = Point3(std::fmin(a.x(), b.x()), std::fmin(a.y(), b.y()), std::fmin(a.z(), b.z()));
    Point3 max = Point3(std::fmax(a.x(), b.x()), std::fmax(a.y(), b.y()), std::fmax(a.z(), b.z()));

    Vec3 dx = Vec3(max.x() -  min.x(), 0, 0);
    Vec3 dy = Vec3(0, max.y() - min.y(), 0);
    Vec3 dz = Vec3(0, 0, max.z() - min.z());

    sides->add(make_shared<Quad>(Point3(min.x(), min.y(), max.z()), dx, dy, mat));
    sides->add(make_shared<Quad>(Point3(max.x(), min.y(), max.z()), -dz,  dy, mat));
    sides->add(make_shared<Quad>(Point3(max.x(), min.y(), min.z()), -dx,  dy, mat));
    sides->add(make_shared<Quad>(Point3(min.x(), min.y(), min.z()),  dz,  dy, mat));
    sides->add(make_shared<Quad>(Point3(min.x(), max.y(), max.z()),  dx, -dz, mat));
    sides->add(make_shared<Quad>(Point3(min.x(), min.y(), min.z()),  dx,  dz, mat));

    return sides;
}

#endif