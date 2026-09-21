#ifndef RAY_H
#define RAY_H

#include "common_constants.h"

/**
* Represents a 3D line in ray tracing.
*/
class Ray {
    private:
        Point3 orig;
        Vec3 dir;
        double tm;
        
    public:
        Ray() {}

        /**
        * Initializes origin, direction, and time for a ray.
        */
        Ray(const Point3& origin, const Vec3& direction, double time) : orig(origin), dir(direction), tm(time) {}

        /**
        * Initializes origin and direction for a ray.
        */
        Ray(const Point3& origin, const Vec3& direction) : Ray(origin, direction, 0) {}

        const Point3& origin() const { return orig; }
        const Vec3& direction() const { return dir; }

        double time() const { return tm; }

        /**
        * Grabs 3D position P(t) along the ray at param distance t.
        */
        Point3 at(double t) const {
            return orig + t * dir;
        }
};

#endif