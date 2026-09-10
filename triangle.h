#ifndef TRIANGLE_H
#define TRIANGLE_H

class Triangle : public Hittable {
    private:
        Point3 v1, v2, v3;
        shared_ptr<Material> mat;
        Aabb bbox;
    public:
        Triangle(const Point3& v1, const Point3& v2, const Point3& v3, shared_ptr<Material> mat) 
        : v1(v1), v2(v2), v3(v3), mat(mat) {}

        /**
        blah

        Majority of the method is taken from chapter 6.5.3 of the PBRT https://pbr-book.org/4ed/Shapes/Triangle_Meshes#RayndashTriangleIntersection
        */
        bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const override {
            // return no intersection if triangle is degenerate (all 3 points align in a line)
            if (length_squared(cross(v3 - v1, v2 - v1)) == 0) {
                return false;
            }

            // transform triangle vertices into ray coordinate space

            // translate vertices based on ray origin
            Point3 v1t = v1 - r.orig;
            Point3 v2t = v2 - r.orig;
            Point3 v3t = v3 - r.orig;

            // permute components of triangle vertices and ray direction
        }
};

#endif