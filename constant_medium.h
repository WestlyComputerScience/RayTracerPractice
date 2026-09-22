#ifndef CONSTANT_MEDIUM_H
#define CONSTANT_MEDIUM_H

#include "hittable.h"
#include "material.h"
#include "texture.h"

/**
* Implements a volumetric object enclosed within a geometric boundary.
*
* Note: This doesn't handle shapes that contain voids (toruses etc.)
*/
class ConstantMedium : public Hittable {
    private:
        shared_ptr<Hittable> boundary;
        double neg_inv_density;
        shared_ptr<Material> phase_function;
    public:
        /**
        * Initializes the volume using a procedural/image-based texture for scattering albedo.
        */
        ConstantMedium(shared_ptr<Hittable> boundary, double density, shared_ptr<Texture> tex) : boundary(boundary), 
            neg_inv_density(-1 / density), phase_function(make_shared<Isotropic>(tex)) {}
        
        /**
        * Initializes the volume using a constant color for scattering albedo.
        */
        ConstantMedium(shared_ptr<Hittable> boundary, double density, const Color& albedo) : boundary(boundary), 
            neg_inv_density(-1 / density), phase_function(make_shared<Isotropic>(albedo)) {}

        /**
        * Determines if a ray scatters against volume particles inside the boundary.
        */
        bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const override {
            HitRecord rec1, rec2;

            // Must be within the boundary
            if (!boundary->hit(r, Interval::universe, rec1)) return false;
            if (!boundary->hit(r, Interval(rec1.t + 0.0001, infinity), rec2)) return false;

            // Clamps entry time
            if (rec1.t < ray_t.min) rec1.t = ray_t.min;
            if (rec2.t > ray_t.max) rec2.t = ray_t.max;

            if (rec1.t >= rec2.t) return false;

            if (rec1.t < 0) rec1.t = 0;

            // Calculates total distance traveled inside the boundary
            double ray_length = r.direction().length();
            double distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
            double hit_distance = neg_inv_density * std::log(random_double());

            // Has chance to completely pass through the volume
            if (hit_distance > distance_inside_boundary) return false;

            // Populate the hit record
            rec.t = rec1.t + hit_distance / ray_length;
            rec.p = r.at(rec.t);
            rec.normal = Vec3(1, 0, 0); // arbitrary
            rec.front_face = true; // arbitrary
            rec.mat = phase_function;

            return true;
        }

        Aabb bounding_box() const override { return boundary->bounding_box(); }
};

#endif