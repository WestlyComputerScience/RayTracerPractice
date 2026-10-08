#ifndef PDF_H
#define PDF_H

#include "hittable_list.h"
#include "onb.h"

class Pdf {
    public:
        virtual ~Pdf() {}

        virtual real value(const Vec3& direction) const = 0;
        virtual Vec3 generate() const = 0;
};

class SpherePdf : public Pdf {
    public:
        SpherePdf() {}

        real value(const Vec3& direction) const override { return 1 / (4 * pi); }

        Vec3 generate() const override { return random_unit_vector(); }
};

class CosinePdf : public Pdf {
    private:
        Onb uvw;
    public:
        CosinePdf(const Vec3& w) : uvw(w) {}

        real value(const Vec3& direction) const override {
            real cosine_theta = dot(unit_vector(direction), uvw.w());
            return std::fmax(0, cosine_theta / pi);
        }

        Vec3 generate() const override {
            return uvw.transform(random_cosine_direction());
        }
};

class HittablePdf : public Pdf {
    private:
        const Hittable& objects;
        Point3 origin;
    public:
        HittablePdf(const Hittable& objects, const Point3& origin) : objects(objects), origin(origin) {}

        real value(const Vec3& direction) const override {
            return objects.pdf_value(origin, direction);
        }

        Vec3 generate() const override { return objects.random(origin); }
};

class MixturePdf : public Pdf {
    private:
        shared_ptr<Pdf> p[2];
    public:
        MixturePdf(shared_ptr<Pdf> p0, shared_ptr<Pdf> p1) {
            p[0] = p0;
            p[1] = p1;
        }

        real value(const Vec3& direction) const override {
            return 0.5 * p[0]->value(direction) + 0.5 * p[1]->value(direction);
        }

        Vec3 generate() const override {
            if (random_real() < 0.5) {
                return p[0]->generate();
            } else {
                return p[1]->generate();
            }
        }
};

#endif