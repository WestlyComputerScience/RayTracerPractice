#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.h"
#include "texture.h"

/**
* Abstract class for the material system.
*/
class Material {
    public:
        virtual ~Material() = default;

        /**
        * Calculates the light emmitted directly by the material at a specific point, black by default.
        */
        virtual Color emitted(double u, double v, const Point3& p) const {
            return Color(0, 0, 0);
        }

        /**
        * Determines how an incoming ray bounces, no bounce by default.
        */
        virtual bool scatter(const Ray& r, const HitRecord& rec, Color& attenuation, Ray& scattered) const {
            return false;
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
        bool scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered) const override {
            Vec3 scatter_direction = rec.normal + random_unit_vector();

            if (scatter_direction.near_zero()) {
                scatter_direction = rec.normal;
            }

            scattered = Ray(rec.p, scatter_direction, r_in.time());
            attenuation = tex->value(rec.u, rec.v, rec.p);
            return true;
        }
};

/**
* Implementation of specular materials (reflective).
*/
class Metal : public Material {
    private:
        Color albedo;
        double fuzz;
    public:
        /**
        * Takes an albedo to store the RGB reflectance color and a fuzziness factor to blur the reflections.
        */
        Metal(const Color& albedo, double fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}

        /**
        * Calculates specular reflection and applies roughness offset to produce outbound rays.
        */
        bool scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered) const override {
            Vec3 reflected = reflect(r_in.direction(), rec.normal);
            reflected = unit_vector(reflected) + (fuzz * random_unit_vector());
            scattered = Ray(rec.p, reflected, r_in.time());
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > 0);
        }   
};

/**
* Implementation of dialectric materials (i.e. glass, water...).
*/
class Dielectric : public Material {
    private:
        // index of refreaction (IOR) the material contains.
        double refraction_index;

        /**
        * Computes the reflectance given the IOR and incident angle.
        */
        static double reflectance(double cosine, double refraction_index) {
            double r0 = (1 - refraction_index) / (1 + refraction_index);
            r0 = r0 * r0;
            return r0 + (1 - r0) * std::pow((1 - cosine), 5);
        }
    public:
        /**
        * Given an objects IOR, creates a dielectric material.
        */
        Dielectric(double refraction_index) : refraction_index(refraction_index) {}

        /**
        * Determines if the incoming ray refracts into/out of the dielectric or if it reflects off the surface boundary.
        */
        bool scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered) const override {
            attenuation = Color(1.0, 1.0, 1.0);
            double ri = rec.front_face ? (1.0 / refraction_index) : refraction_index;

            Vec3 unit_direction = unit_vector(r_in.direction());
            double cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0);
            double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);

            bool cannot_refract = ri * sin_theta > 1.0;
            Vec3 direction;
            if (cannot_refract || reflectance(cos_theta, ri) > random_double()) {
                direction = reflect(unit_direction, rec.normal);
            } else {
                direction = refract(unit_direction, rec.normal, ri);
            }

            scattered = Ray(rec.p, direction, r_in.time());
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
        Color emitted(double u, double v, const Point3& p) const override {
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
        bool scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered) const override {
            scattered = Ray(rec.p, random_unit_vector(), r_in.time());
            attenuation = tex->value(rec.u, rec.v, rec.p);
            return true;
        }
};

#endif