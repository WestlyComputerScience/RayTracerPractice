#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "common_constants.h"

class Triangle : public Hittable {
    private:
        Point3 v1, v2, v3;
        shared_ptr<Material> mat;
        Aabb bbox;
    public:
        Triangle(const Point3& v1, const Point3& v2, const Point3& v3, shared_ptr<Material> mat) 
        : v1(v1), v2(v2), v3(v3), mat(mat) {
            Aabb box_v1_v2(v1, v2);
            Aabb box_v3(v3, v3);
            bbox = Aabb(box_v1_v2, box_v3);
        }

        /**
        blah

        Majority of the method is taken from chapter 6.5.3 of the PBRT https://pbr-book.org/4ed/Shapes/Triangle_Meshes#RayndashTriangleIntersection
        */
        bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const override {
            // return no intersection if triangle is degenerate (all 3 points align in a line)
            Vec3 outward_normal = cross(v2 - v1, v3 - v1);
            if (outward_normal.length_squared() == 0) return false;

            // make all the ray origins sit at (0, 0, 0)
            Point3 v1t = v1 - r.origin();
            Point3 v2t = v2 - r.origin();
            Point3 v3t = v3 - r.origin();

            // permute components to get the z dimension to have the absolute value of the ray's direction to be the largest
            Vec3 r_dir = r.direction();
            int kz = max_component_index(abs(r_dir));
            int kx = kz + 1;
            if (kx == 3) kx = 0;
            int ky = kx + 1;
            if (ky == 3) ky = 0;
            
            Vec3 d = permute(r_dir, {kx, ky, kz});
            v1t = permute(v1t, {kx, ky, kz});
            v2t = permute(v2t, {kx, ky, kz});
            v3t = permute(v3t, {kx, ky, kz});

            // apply a shear transformation to translate vertex to the +z axis
            double sx = -d.x() / d.z();
            double sy = -d.y() / d.z();
            double sz = 1.0 / d.z();
            v1t[0] += sx * v1t[2];
            v1t[1] += sy * v1t[2];
            v2t[0] += sx * v2t[2];
            v2t[1] += sy * v2t[2];
            v3t[0] += sx * v3t[2];
            v3t[1] += sy * v3t[2];

            // Looking at the triangle from a 2D view now, 
            double e0 = difference_of_products(v2t.x(), v3t.y(), v2t.y(), v3t.x());
            double e1 = difference_of_products(v3t.x(), v1t.y(), v3t.y(), v1t.x());
            double e2 = difference_of_products(v1t.x(), v2t.y(), v1t.y(), v2t.x());

            // if all 3 vallues share the same sign, (0, 0) is inside the triangle. If any sign differs, the ray missed.
            if ((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0)) return false;
            double det = e0 + e1 + e2;
            if (det == 0) return false;

            // Get the scaled hit distance to triangle and test against ray_t range
            v1t[2] *= sz;
            v2t[2] *= sz;
            v3t[2] *= sz;

            double tScaled = e0 * v1t.z() + e1 * v2t.z() + e2 * v3t.z();
            if (det < 0) {
                if (tScaled > ray_t.min * det || tScaled < ray_t.max * det) return false;
            } else {
                if (tScaled < ray_t.min * det || tScaled > ray_t.max * det) return false;
            }

            // Compute baycentric coords and t value for triangle intersection
            double invDet = 1 / det;
            double b1 = e1 * invDet;
            double b2 = e2 * invDet;
            double t = tScaled * invDet;

            // populate HitRecord
            rec.t = t;
            rec.p = r.at(t);
            rec.u = b1;
            rec.v = b2;
            rec.mat = mat;
            rec.set_face_normal(r, unit_vector(outward_normal));
            return true;
        }

        Aabb bounding_box() const override { return bbox; }
};

#endif