#ifndef AABB_H
#define AABB_H


/**
* This is referred to as "axis-aligned bounding rectangular parallelepipeds" or, in short, "axis-aligned bounding boxes."
* It's goal is to represent a 3D bounding volume and form the foundation for spatial acceleration for BvhNode.
*/
class Aabb {
    private:
        /**
        * Checks if the size of an interval is less than 0.0001
        */
        void pad_to_minimums() {
            double delta = 0.0001;
            if (x.size() < delta) x = x.expand(delta);
            if (y.size() < delta) y = y.expand(delta);
            if (z.size() < delta) z = z.expand(delta);
        }
    public:
        Interval x, y, z;
        
        static const Aabb empty, universe;

        Aabb() {}

        /**
        * Constructs a box directly from 1D intervals and applies padding.
        */
        Aabb(const Interval& x, const Interval& y, const Interval& z) : x(x), y(y), z(z) {
            pad_to_minimums();
        }

        /**
        * Constructs a box enclosing 2 3D corner points and applies padding.
        */
        Aabb(const Point3& a, const Point3& b) {
            x = (a[0] <= b[0]) ? Interval(a[0], b[0]) : Interval(b[0], a[0]);
            y = (a[1] <= b[1]) ? Interval(a[1], b[1]) : Interval(b[1], a[1]);
            z = (a[2] <= b[2]) ? Interval(a[2], b[2]) : Interval(b[2], a[2]);

            pad_to_minimums();
        }

        /**
        * Constructs a merged bounding box given 2 bounding boxes.
        */
        Aabb(const Aabb& box0, const Aabb& box1) {
            x = Interval(box0.x, box1.x);
            y = Interval(box0.y, box1.y);
            z = Interval(box0.z, box1.z);
        }

        /**
        * Index-based lookup giving a reference to the interval.
        */
        const Interval& axis_interval(int n) const {
            if (n == 1) return y;
            if (n == 2) return z;
            return x;
        }

        /**
        * Evaluates a ray-Aabb intersection using a slab method.
        */
        bool hit(const Ray& r, Interval ray_t) const {
            const Point3& ray_orig = r.origin();
            const Vec3& ray_dir = r.direction();

            for (int axis = 0; axis < 3; axis++) {
                // Grab inverse ray direction.
                const Interval& ax = axis_interval(axis);
                const double adinv = 1.0 / ray_dir[axis];

                // Calculates parametric ray intersection distances.
                double t0 = (ax.min - ray_orig[axis]) * adinv;
                double t1 = (ax.max - ray_orig[axis]) * adinv;

                // Narrows down the parametric interval.
                if (t0 < t1) {
                    if (t0 > ray_t.min) ray_t.min = t0;
                    if (t1 < ray_t.max) ray_t.max = t1;
                } else {
                    if (t1 > ray_t.min) ray_t.min = t1;
                    if (t0 < ray_t.max) ray_t.max = t0;
                }

                // returns false if ray missed the volume overlap region.
                if (ray_t.max <= ray_t.min) return false;
            }
            return true;
        }

        /**
        * Returns the index of the axis with the largest interval span.
        */
        int longest_axis() const {
            if (x.size() > y.size()) {
                return x.size() > z.size() ? 0 : 2;
            } else {
                return y.size() > z.size() ? 1 : 2;
            }
        }
};

const Aabb Aabb::empty = Aabb(Interval::empty, Interval::empty, Interval::empty);
const Aabb Aabb::universe = Aabb(Interval::universe, Interval::universe, Interval::universe);

/**
* Returns a new Aabb translated by vector offset.
*/
Aabb operator+(const Aabb& bbox, const Vec3& offset) {
    return Aabb(bbox.x + offset.x(), bbox.y + offset.y(), bbox.z + offset.z());
}

Aabb operator+(const Vec3& offset, const Aabb& bbox) {
    return bbox + offset;
}

#endif