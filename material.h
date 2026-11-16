#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.h"
#include "pdf.h"
#include "texture.h"

class ScatterRecord {
    public:
        Color attenuation;
        shared_ptr<Pdf> pdf_ptr;
        bool skip_pdf;
        Ray skip_pdf_ray;
};

/**
* Abstract class for the material system.
*/
class Material {
    public:
        virtual ~Material() = default;

        /**
        * Calculates the light emmitted directly by the material at a specific point, black by default.
        */
        virtual Color emitted(const Ray& r_in, const HitRecord& rec, real u, real v, const Point3& p) const {
            return Color(0, 0, 0);
        }

        /**
        * Determines how an incoming ray bounces, no bounce by default.
        */
        virtual bool scatter(const Ray& r, const HitRecord& rec, ScatterRecord& srec) const {
            return false;
        }

        virtual real scattering_pdf(const Ray& r_in, const HitRecord& rec, const Ray& scattered) const {
            return 0;
        }

        virtual Color scattering_value(
            const Ray& r_in,
            const HitRecord& rec,
            const Ray& scattered,
            const Color& attenuation
        ) const {
            return attenuation * scattering_pdf(r_in, rec, scattered);
        }
};

/**
* Implementation of the diffuse material (i.e. a surface that scatters light uniformly in all directions).
*/
class Lambertian : public Material {
    private:
        shared_ptr<Texture> tex;
    public:
        /**
        * Wraps a single solid color texture.
        */
        Lambertian(const Color& albedo) : tex(make_shared<SolidColor>(albedo)) {}

        /**
        * Accepts a custom texture.
        */
        Lambertian(shared_ptr<Texture> tex) : tex(tex) {}

        /**
        * Computes the out ray direction and surface color attenuation.
        */
        bool scatter(const Ray& r_in, const HitRecord& rec, ScatterRecord& srec) const override {
            srec.attenuation = tex->value(rec.u, rec.v, rec.p);
            srec.pdf_ptr = make_shared<CosinePdf>(rec.normal);
            srec.skip_pdf = false;
            return true;
        }

        real scattering_pdf(const Ray& r_in, const HitRecord& rec, const Ray& scattered) const override {
            real cos_theta = dot(rec.normal, unit_vector(scattered.direction()));
            return cos_theta < 0 ? 0 : cos_theta / pi;
        }
};

/**
* Implementation of specular materials (reflective).
*/
class Metal : public Material {
    private:
        Color albedo;
        real fuzz;
    public:
        /**
        * Takes an albedo to store the RGB reflectance color and a fuzziness factor to blur the reflections.
        */
        Metal(const Color& albedo, real fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}

        /**
        * Calculates specular reflection and applies roughness offset to produce outbound rays.
        */
        bool scatter(const Ray& r_in, const HitRecord& rec, ScatterRecord& srec) const override {
            Vec3 reflected = reflect(r_in.direction(), rec.normal);
            reflected = unit_vector(reflected) + (fuzz * random_unit_vector());
            
            srec.attenuation = albedo;
            srec.pdf_ptr = nullptr;
            srec.skip_pdf = true;
            srec.skip_pdf_ray = Ray(rec.p, reflected, r_in.time());

            return true;
        }   
};

/**
* Implementation of dialectric materials (i.e. glass, water...).
*/
class Dielectric : public Material {
    private:
        // index of refreaction (IOR) the material contains.
        real refraction_index;

        /**
        * Computes the reflectance given the IOR and incident angle.
        */
        static real reflectance(real cosine, real refraction_index) {
            real r0 = (1 - refraction_index) / (1 + refraction_index);
            r0 = r0 * r0;
            return r0 + (1 - r0) * std::pow((1 - cosine), 5);
        }
    public:
        /**
        * Given an objects IOR, creates a dielectric material.
        */
        Dielectric(real refraction_index) : refraction_index(refraction_index) {}

        /**
        * Determines if the incoming ray refracts into/out of the dielectric or if it reflects off the surface boundary.
        */
        bool scatter(const Ray& r_in, const HitRecord& rec, ScatterRecord& srec) const override {
            srec.attenuation = Color(1.0, 1.0, 1.0);
            srec.pdf_ptr = nullptr;
            srec.skip_pdf = true;
            real ri = rec.front_face ? (1.0 / refraction_index) : refraction_index;

            Vec3 unit_direction = unit_vector(r_in.direction());
            real cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0);
            real sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);

            bool cannot_refract = ri * sin_theta > 1.0;
            Vec3 direction;
            if (cannot_refract || reflectance(cos_theta, ri) > random_real()) {
                direction = reflect(unit_direction, rec.normal);
            } else {
                direction = refract(unit_direction, rec.normal, ri);
            }

            srec.skip_pdf_ray = Ray(rec.p, direction, r_in.time());
            return true;
        }
};

/**
* Implementation of an emissive material to turn objects into light sources.
*/
class DiffuseLight : public Material {
    private:
        shared_ptr<Texture> tex;
    public:
        /**
        * Holds the emission texture sampled to determine the emitted light color and energy.
        */
        DiffuseLight(shared_ptr<Texture> tex) : tex(tex) {}

        /**
        * Wraps a solid RGB color into a solid color texture.
        */
        DiffuseLight(const Color& emit) : tex(make_shared<SolidColor>(emit)) {}

        /**
        * Calculates the light spectrum and intensity at a given point.
        */
        Color emitted(const Ray& r_in, const HitRecord& rec, real u, real v, const Point3& p) const override {
            if (!rec.front_face) return Color(0, 0, 0);
            return tex->value(u, v, p);
        }
};

/**
* Implements a uniform scattering material (i.e. fog, smoke, etc.).
*/
class Isotropic : public Material {
    private:
        shared_ptr<Texture> tex;
    public:
        /**
        * Wraps solid RGB color into a solid color texture.
        */
        Isotropic(const Color& albedo) : tex(make_shared<SolidColor>(albedo)) {}

        /**
        * Flexible constructor that accepts any texture.
        */
        Isotropic(shared_ptr<Texture> tex) : tex(tex) {}

        /**
        * Computes a random outbound ray direction and sets the color attenuation.
        */
        bool scatter(const Ray& r_in, const HitRecord& rec, ScatterRecord& srec) const override {
            srec.attenuation = tex->value(rec.u, rec.v, rec.p);
            srec.pdf_ptr = make_shared<SpherePdf>();
            srec.skip_pdf = false;
            return true;
        }

        real scattering_pdf(const Ray& r_in, const HitRecord& rec, const Ray& scattered) const override { return 1 / (4 * pi); }
};

/**
 * Implementation of a Cook-Torrance PBR material
 * using the GGX microfacet distribution.
 */
class GGXMaterial : public Material {
private:
    shared_ptr<Texture> albedo;
    real roughness;
    real metalness;

    Color fresnel_schlick(real cos_theta, const Color& f0) const {
        real t = std::pow(
            std::clamp(1.0 - cos_theta, 0.0, 1.0),
            5.0
        );

        return Color(
            f0.r() + (1.0 - f0.r()) * t,
            f0.g() + (1.0 - f0.g()) * t,
            f0.b() + (1.0 - f0.b()) * t
        );
    }

    real distribution_ggx(
        const Vec3& n,
        const Vec3& h,
        real roughness
    ) const {
        real alpha = roughness * roughness;
        real alpha2 = alpha * alpha;

        real n_dot_h = std::max(dot(n, h), real(0));
        real n_dot_h2 = n_dot_h * n_dot_h;

        real denom = n_dot_h2 * (alpha2 - 1.0) + 1.0;

        return alpha2 / (pi * denom * denom);
    }

    real geometry_smith_g1(
        real n_dot_v,
        real roughness
    ) const {
        real alpha = roughness * roughness;
        real alpha2 = alpha * alpha;

        real denom = n_dot_v + std::sqrt(
            alpha2 + (1.0 - alpha2) * n_dot_v * n_dot_v
        );

        return (2.0 * n_dot_v) / std::max(denom, real(1e-8));
    }

    real geometry_smith(
        const Vec3& n,
        const Vec3& v,
        const Vec3& l,
        real roughness
    ) const {
        real n_dot_v = std::max(dot(n, v), real(0));
        real n_dot_l = std::max(dot(n, l), real(0));

        return geometry_smith_g1(n_dot_v, roughness) *
            geometry_smith_g1(n_dot_l, roughness);
    }

    /**
     * PDF of the reflected direction when sampling
     * halfway vectors using the GGX NDF.
     */
    real ggx_pdf(
        const Vec3& n,
        const Vec3& v,
        const Vec3& h,
        real roughness
    ) const {
        real n_dot_v = std::max(dot(n, v), real(0));
        real n_dot_h = std::max(dot(n, h), real(0));
        real v_dot_h = dot(v, h);

        if (n_dot_v <= 1e-8 ||
            n_dot_h <= 0.0 ||
            v_dot_h <= 1e-8)
            return 0.0;

        real D = distribution_ggx(n, h, roughness);
        real G1 = geometry_smith_g1(n_dot_v, roughness);

        // VNDF halfway-vector PDF transformed to reflected direction.
        return D * G1 / (4.0 * n_dot_v);
    }

public:
    GGXMaterial(
        const Color& albedo,
        real roughness,
        real metalness
    )
        : albedo(make_shared<SolidColor>(albedo)),
          roughness(std::clamp(roughness, real(0.001), real(1.0))),
          metalness(std::clamp(metalness, real(0.0), real(1.0))) {}

    GGXMaterial(
        shared_ptr<Texture> tex,
        real roughness,
        real metalness
    )
        : albedo(tex),
          roughness(std::clamp(roughness, real(0.001), real(1.0))),
          metalness(std::clamp(metalness, real(0.0), real(1.0))) {}

    /**
     * Samples and evaluates the Cook-Torrance GGX
     * specular BRDF.
     */
    bool scatter(
    const Ray& r_in,
    const HitRecord& rec,
    ScatterRecord& srec
) const override {

    Vec3 v = -unit_vector(r_in.direction());

    real n_dot_v = dot(rec.normal, v);

    if (n_dot_v <= 0.0)
        return false;

    // Calculate base reflectivity.
    Color base_color = albedo->value(rec.u, rec.v, rec.p);

    Color f0(
        0.04 * (1.0 - metalness) + base_color.r() * metalness,
        0.04 * (1.0 - metalness) + base_color.g() * metalness,
        0.04 * (1.0 - metalness) + base_color.b() * metalness
    );

    // Store the material parameters for BRDF evaluation.
    srec.attenuation = f0;

    // Provide the GGX sampling distribution.
    srec.pdf_ptr = make_shared<GGXPdf>(
        rec.normal,
        v,
        roughness
    );

    srec.skip_pdf = false;

    return true;
}

Color scattering_value(
    const Ray& r_in,
    const HitRecord& rec,
    const Ray& scattered,
    const Color& attenuation
) const override {

    Vec3 n = rec.normal;
    Vec3 v = -unit_vector(r_in.direction());
    Vec3 l = unit_vector(scattered.direction());

    real n_dot_v = std::max(dot(n, v), real(0));
    real n_dot_l = std::max(dot(n, l), real(0));

    if (n_dot_v <= 1e-8 || n_dot_l <= 0.0)
        return Color(0, 0, 0);

    Vec3 h = unit_vector(v + l);

    Color base_color = albedo->value(rec.u, rec.v, rec.p);

    Color f0(
        0.04 * (1.0 - metalness) + base_color.r() * metalness,
        0.04 * (1.0 - metalness) + base_color.g() * metalness,
        0.04 * (1.0 - metalness) + base_color.b() * metalness
    );

    Color F = fresnel_schlick(
        std::max(dot(v, h), real(0)),
        f0
    );

    real D = distribution_ggx(n, h, roughness);
    real G = geometry_smith(n, v, l, roughness);

    real factor = D * G / (4.0 * n_dot_v);

    return F * factor;
}
};


#endif