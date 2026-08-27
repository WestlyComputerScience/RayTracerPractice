#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include <vector>
#include "common_constants.h"
#include "hittable.h"

using std::make_shared;
using std::shared_ptr;

class HittableList : public Hittable {
    public:
        std::vector<shared_ptr<Hittable>> objects;

        HittableList() {}
        HittableList(shared_ptr<Hittable> object) { add(object); }

        void clear() { objects.clear(); }

        void add(shared_ptr<Hittable> object) {
            objects.push_back(object);
        }

        bool hit(const Ray& r, double ray_tmin, double ray_tmax, HitRecord& rec) const override {
            HitRecord temp_rec;
            bool hit_anything = false;
            double closest_so_far = ray_tmax;

            for (const shared_ptr<Hittable>& object : objects) {
                if (object->hit(r, ray_tmin, closest_so_far, temp_rec)) {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }

            return hit_anything;
        }
};

#endif