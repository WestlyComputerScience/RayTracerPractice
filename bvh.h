#ifndef BVH_H
#define BVH_H

#include <algorithm>

#include "aabb.h"
#include "hittable.h"
#include "hittable_list.h"

class BvhNode : public Hittable {
    private:
        shared_ptr<Hittable> left;
        shared_ptr<Hittable> right;
        Aabb bbox;

        static bool box_compare(const shared_ptr<Hittable> a, const shared_ptr<Hittable> b, int axis_index) {
            Interval a_axis_interval = a->bounding_box().axis_interval(axis_index);
            Interval b_axis_interval = b->bounding_box().axis_interval(axis_index);
            return a_axis_interval.min < b_axis_interval.min;
        }

        static bool box_x_compare(const shared_ptr<Hittable> a, const shared_ptr<Hittable> b) {
            return box_compare(a, b, 0);
        }

        static bool box_y_compare(const shared_ptr<Hittable> a, const shared_ptr<Hittable> b) {
            return box_compare(a, b, 1);
        }

        static bool box_z_compare(const shared_ptr<Hittable> a, const shared_ptr<Hittable> b) {
            return box_compare(a, b, 2);
        }

    public:
        BvhNode(HittableList list) : BvhNode(list.objects, 0, list.objects.size()) {
            // Small C++ thing here to note. The constructor (without span indices) will create an implicit copy
            // of the hittable list, which we'll modify. The lifetime of this copy extends only until the constructor exits.
            // For this implementation, it's ok since we only need to persist the resulting bounding volume hierarchy.
        }

        BvhNode(std::vector<shared_ptr<Hittable>>& objects, size_t start, size_t end) { 
            bbox = Aabb::empty;
            for (size_t object_index = start; object_index < end; object_index++) {
                bbox = Aabb(bbox, objects[object_index]->bounding_box());
            }

            int axis = bbox.longest_axis();

            auto comparator = (axis == 0) ? box_x_compare
                : (axis == 1) ? box_y_compare
                              : box_z_compare;

            size_t object_span = end - start;

            if (object_span == 1) {
                left = right = objects[start];
            } else if (object_span == 2) {
                left = objects[start];
                right = objects[start + 1];
            } else {
                std::sort(std::begin(objects) + start, std::begin(objects) + end, comparator);

                double mid = start + object_span / 2;
                left = make_shared<BvhNode>(objects, start, mid);
                right = make_shared<BvhNode>(objects, mid, end);
            }
        }

        bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const override {
            if (!bbox.hit(r, ray_t)) return false;

            bool hit_left = left->hit(r, ray_t, rec);
            bool hit_right = right->hit(r, Interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec);

            return hit_left || hit_right;
        }

        Aabb bounding_box() const override { return bbox; }
};

#endif