#ifndef HITTABLE_H
#define HITTABLE_H

#include "aabb.h"
#include "common_constants.h"

class Material;

/**
* A temporary container for collision details. Contains information like point, normal, material, ray paramater distance (t),
* texture coords (u, v), and if the ray struct the exterior or interior surface.
*/
class HitRecord {
    public:
        Point3 p;
        Vec3 normal;
        shared_ptr<Material> mat;
        double t;
        double u, v;
        bool front_face;

        /**
        * Standardizes normal orientation. If dot product < 0, ray is outside object; otherwise, it's inside.
        */
        void set_face_normal(const Ray& r, const Vec3& outward_normal) {
            front_face = dot(r.direction(), outward_normal) < 0;
            normal = front_face ? outward_normal : -outward_normal;
        }
};

/**
* Abstract base interface for every object in the ray tracer that can intersect. Defines how objects are hit and 
* how the object is bounded. 
*/
class Hittable {
    public:
        virtual ~Hittable() = default;

        /**
        * Determines if a ray intersects an object.
        */
        virtual bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const = 0;

        /**
        * Returns an Axis-Aligned Bounding Box enclosing the object, used for Bvhs.
        */
        virtual Aabb bounding_box() const = 0;
};

/**
* A decorator that translates any Hittable object.
*/
class Translate : public Hittable {
    private:
        shared_ptr<Hittable> object;
        Vec3 offset;
        Aabb bbox;
    public:
        /**
        * Translates an object in 3D space by an offset vector.
        */
        Translate(shared_ptr<Hittable> object, const Vec3& offset) : object(object), offset(offset) {
            bbox = object->bounding_box() + offset;
        }

        /**
        * Calculates if a ray hit the translated object.
        */
        bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const override {
            // move ray backwards by offset
            Ray offset_r(r.origin() - offset, r.direction(), r.time());

            // determine if the intersection exists along the offset ray
            if (!object->hit(offset_r, ray_t, rec)) return false;

            // move intersection point forwards by the offset
            rec.p += offset;
            return true;
        }

        Aabb bounding_box() const override { return bbox; }
};

/**
* A decorator that rotates any hittable object. The idea is to rotate a point/vector by theta around the Y-axis with matrix multiplication.
* Then, to rotate by +theta, we transform the incoming ray by the inverse rotation -theta (below is a visual representation for myself in debugging).
* 
* Initially:
* ---------------------------
* | cos(theta) 0 sin(theta) |
* |     0      1       0    |
* |-sin(theta) 0 cos(theta) |
* ---------------------------
* Rotated:
* ---------------------------
* | cos(theta) 0 -sin(theta)|
* |     0      1       0    |
* | sin(theta) 0 cos(theta) |
* ---------------------------
*/
class RotateY : public Hittable {
    private:
        shared_ptr<Hittable> object;
        double sin_theta;
        double cos_theta;
        Aabb bbox;
    public:
        /**
        * Rotates an object around the Y-axis by angle theta.
        */
        RotateY(shared_ptr<Hittable> object, double angle) : object(object) {
            // store original bounds
            double radians = degrees_to_radians(angle);
            sin_theta = std::sin(radians);
            cos_theta = std::cos(radians);
            bbox = object->bounding_box();

            Point3 min(infinity, infinity, infinity);
            Point3 max(-infinity, -infinity, -infinity);

            // iterating all 8 corners of initial bounding box
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    for (int k = 0; k < 2; k++) {
                        double x = i * bbox.x.max + (1 - i) * bbox.x.min;
                        double y = j * bbox.y.max + (1 - j) * bbox.y.min;
                        double z = k * bbox.z.max + (1 - k) * bbox.z.min;

                        double newX = cos_theta * x + sin_theta * z;
                        double newZ = -sin_theta * x + cos_theta * z;

                        Vec3 tester(newX, y, newZ);

                        // rotating each corner and refitting the bounding box
                        for (int c = 0; c < 3; c++) {
                            min[c] = std::fmin(min[c], tester[c]);
                            max[c] = std::fmax(max[c], tester[c]);
                        }
                    }
                }
            }

            bbox = Aabb(min, max);
        }

        /**
        * Calculates if a ray hit the rotated object.
        */
        bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const override {
            // transform the ray from world space to object space
            Point3 origin = Point3(
                (cos_theta * r.origin().x()) - (sin_theta * r.origin().z()),
                r.origin().y(),
                (sin_theta * r.origin().x()) + (cos_theta * r.origin().z())
            );

            Vec3 direction = Vec3(
                (cos_theta * r.direction().x()) - (sin_theta * r.direction().z()),
                r.direction().y(),
                (sin_theta * r.direction().x()) + (cos_theta * r.direction().z())
            );

            // apply inverse rotation
            Ray rotated_r(origin, direction, r.time());

            if (!object->hit(rotated_r, ray_t, rec)) return false; // if intsersection doesn't exist in object space

            // transform the interscection from object space back to world space
            rec.p = Point3(
                (cos_theta * rec.p.x()) + (sin_theta * rec.p.z()),
                rec.p.y(),
                (-sin_theta * rec.p.x()) + (cos_theta * rec.p.z())
            );

            rec.normal = Vec3(
                (cos_theta * rec.normal.x()) + (sin_theta * rec.normal.z()),
                rec.normal.y(),
                (-sin_theta * rec.normal.x()) + (cos_theta * rec.normal.z())
            );

            return true;
        }
        
        Aabb bounding_box() const override { return bbox; }
};

#endif