#ifndef QUAD_H
#define QUAD_H

class Quad : public Hittable {
    private:
        Point3 Q;
        Vec3 u, v;
        shared_ptr<Material> mat;
        Aabb bbox;
    public:
        Quad(const Point3& Q, const Vec3& u, const Vec3& v, shared_ptr<Material> mat) : Q(Q), u(u), v(v), mat(mat) {
            set_bounding_box();
        }

        virtual void set_bounding_box() {
            Aabb bbox_diagonal1 = Aabb(Q, Q + u + v);
            Aabb bbox_diagonal2 = Aabb(Q + u, Q + v);
            bbox = Aabb(bbox_diagonal1, bbox_diagonal2);
        }

        Aabb bounding_box() const override { return bbox; }

        bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const override {
            return false;
        }
}

#endif