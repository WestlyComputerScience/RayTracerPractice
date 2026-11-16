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

        Vec3 generate(bool& sampled_first) const {
            sampled_first = random_real() < 0.5;
            return p[sampled_first ? 0 : 1]->generate();
        }

        Vec3 generate() const override {
            bool sampled_first;
            return generate(sampled_first);
        }
};

inline Vec3 sample_ggx_vndf(
    const Vec3& view_local,
    real roughness,
    real r1,
    real r2
) {
    real alpha = roughness * roughness;

    // Stretch the view direction.
    Vec3 v = unit_vector(Vec3(
        alpha * view_local.x(),
        alpha * view_local.y(),
        view_local.z()
    ));

    // Construct an orthonormal basis around the stretched view.
    real lensq = v.x() * v.x() + v.y() * v.y();

    Vec3 t1 = lensq > 1e-8
        ? Vec3(-v.y(), v.x(), 0.0) / std::sqrt(lensq)
        : Vec3(1, 0, 0);

    Vec3 t2 = cross(v, t1);

    // Sample a point on the projected disk.
    real radius = std::sqrt(r1);
    real phi = 2.0 * pi * r2;

    real p1 = radius * std::cos(phi);
    real p2 = radius * std::sin(phi);

    // Warp the second coordinate.
    real s = 0.5 * (1.0 + v.z());

    p2 = (1.0 - s) *
         std::sqrt(std::max(0.0, 1.0 - p1 * p1))
         + s * p2;

    // Reconstruct the visible normal.
    real p3 = std::sqrt(
        std::max(0.0, 1.0 - p1 * p1 - p2 * p2)
    );

    Vec3 normal = t1 * p1 + t2 * p2 + v * p3;

    // Unstretch the normal.
    return unit_vector(Vec3(
        alpha * normal.x(),
        alpha * normal.y(),
        std::max(0.0, normal.z())
    ));
}

class GGXPdf : public Pdf {
private:
    Vec3 normal;
    Vec3 view;
    Vec3 tangent;
    Vec3 bitangent;
    real roughness;

    real distribution_ggx(const Vec3& h) const {
        real alpha = roughness * roughness;
        real alpha2 = alpha * alpha;

        real n_dot_h = std::max(dot(normal, h), 0.0);
        real n_dot_h2 = n_dot_h * n_dot_h;

        real denom = n_dot_h2 * (alpha2 - 1.0) + 1.0;

        return alpha2 / (pi * denom * denom);
    }

    real geometry_smith_g1(real n_dot_v) const {
        real alpha = roughness * roughness;
        real alpha2 = alpha * alpha;

        real denom = n_dot_v + std::sqrt(
            alpha2 + (1.0 - alpha2) * n_dot_v * n_dot_v
        );

        return (2.0 * n_dot_v) / std::max(denom, 1e-8);
    }

public:
    GGXPdf(
        const Vec3& normal,
        const Vec3& view,
        real roughness
    )
        : normal(unit_vector(normal)),
          view(unit_vector(view)),
          roughness(roughness) {

        tangent = unit_vector(
            cross(
                std::fabs(normal.x()) > 0.9
                    ? Vec3(0, 1, 0)
                    : Vec3(1, 0, 0),
                normal
            )
        );

        bitangent = cross(normal, tangent);
    }

    real value(const Vec3& direction) const override {
        Vec3 l = unit_vector(direction);

        real n_dot_v = dot(normal, view);
        real n_dot_l = dot(normal, l);

        if (n_dot_v <= 1e-8 || n_dot_l <= 0.0)
            return 0.0;

        Vec3 h = unit_vector(view + l);

        real v_dot_h = dot(view, h);

        if (v_dot_h <= 1e-8)
            return 0.0;

        real D = distribution_ggx(h);
        real G1 = geometry_smith_g1(n_dot_v);

        return D * G1 / (4.0 * n_dot_v);
    }

    Vec3 generate() const override {
        Vec3 view_local(
            dot(view, tangent),
            dot(view, bitangent),
            dot(view, normal)
        );

        Vec3 h_local = sample_ggx_vndf(
            view_local,
            roughness,
            random_real(),
            random_real()
        );

        Vec3 h = unit_vector(
            tangent * h_local.x() +
            bitangent * h_local.y() +
            normal * h_local.z()
        );

        return reflect(-view, h);
    }
};



inline void test_ggx_pdf_normalization() {
    const int theta_steps = 1000;
    const int phi_steps = 2000;

    const real roughness = 0.2;

    // Grazing-angle view direction.
    Vec3 normal(0, 1, 0);
    Vec3 view = unit_vector(Vec3(0.9, 0.2, -0.2));

    GGXPdf pdf(normal, view, roughness);

    real theta_step = (pi / 2.0) / theta_steps;
    real phi_step = (2.0 * pi) / phi_steps;

    real integral = 0.0;

    for (int i = 0; i < theta_steps; ++i) {
        real theta = (i + 0.5) * theta_step;
        real sin_theta = std::sin(theta);
        real cos_theta = std::cos(theta);

        for (int j = 0; j < phi_steps; ++j) {
            real phi = (j + 0.5) * phi_step;

            Vec3 direction(
                sin_theta * std::cos(phi),
                cos_theta,
                sin_theta * std::sin(phi)
            );

            integral += pdf.value(direction)
                * sin_theta
                * theta_step
                * phi_step;
        }
    }

    std::cout << "GGX PDF hemisphere integral = "
              << integral << '\n';
}



#endif