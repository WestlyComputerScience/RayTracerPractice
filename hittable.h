#ifndef HITTABLE_H
#define HITTABLE_H

#include "aabb.h"
#include "common_constants.h"

class Material;

class HitRecord {
    public:
        Point3 p;
        Vec3 normal;
        shared_ptr<Material> mat;
        double t;
        double u, v;
        bool front_face;

        void set_face_normal(const Ray& r, const Vec3& outward_normal) {
            front_face = dot(r.direction(), outward_normal) < 0;
            normal = front_face ? outward_normal : -outward_normal;
        }
};

class Hittable {
    public:
        virtual ~Hittable() = default;

        virtual bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const = 0;

        virtual Aabb bounding_box() const = 0;
};

class Translate : public Hittable {
    private:
        shared_ptr<Hittable> object;
        Vec3 offset;
        Aabb bbox;
    public:
        Translate(shared_ptr<Hittable> object, const Vec3& offset) : object(object), offset(offset) {
            bbox = object->bounding_box() + offset;
        }

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

#endif