#ifndef TEXTURE_H
#define TEXTURE_H

#include "perlin.h"
#include "rtw_stb_image.h"

/**
* Abstract class for all surface patterns in objects.
*/
class Texture {
    public:
        virtual ~Texture() = default;

        /**
        * Calculates RGB color based on a given location.
        */
        virtual Color value(double u, double v, const Point3& p) const = 0;
};

/**
* A texture that represents uniform, constant color across a surface.
*/
class SolidColor : public Texture {
    private:
        Color albedo;
    public:
        /**
        * Stores a constant RGB value to apply.
        */
        SolidColor(const Color& albedo) : albedo(albedo) {}

        /**
        * Stores a RGB value to apply.
        */
        SolidColor(double red, double green, double blue) : SolidColor(Color(red, green, blue)) {}

        /**
        * Applies a constant RGB value to a point.
        */
        Color value(double u, double v, const Point3& p) const override { return albedo; }
};

/**
* A texture that implements a checker-like pattern.
*/
class CheckerTexture : public Texture {
    private:
        double inv_scale;
        shared_ptr<Texture> even;
        shared_ptr<Texture> odd;
    public:
        /**
        * Given custom texture pointers, it applies the checker pattern.
        */
        CheckerTexture(double scale, shared_ptr<Texture> even, shared_ptr<Texture> odd)
        : inv_scale(1.0 / scale), even(even), odd(odd) {}

        /**
        * Wraps 2 solid color params into a solid color texture.
        */
        CheckerTexture(double scale, const Color& c1, const Color& c2) 
        : CheckerTexture(scale, make_shared<SolidColor>(c1), make_shared<SolidColor>(c2)) {}

        /**
        * Computes which texture to apply to a given point for checker-style.
        */
        Color value(double u, double v, const Point3& p) const override {
            int xInt = int(std::floor(inv_scale * p.x()));
            int yInt = int(std::floor(inv_scale * p.y()));
            int zInt = int(std::floor(inv_scale * p.z()));

            bool isEven = (xInt + yInt + zInt) % 2 == 0;

            return isEven ? even->value(u, v, p) : odd->value(u, v, p);
        }
};

/**
* A texture that maps a 2D image onto surfaces.
*/
class ImageTexture : public Texture {
    private:
        RtwImage image; // Rtw is the image library of choice
    public:
        /**
        * Takes an image to construct the texture.
        */
        ImageTexture(const char* filename) : image(filename) {}

        /**
        * Samples the underlying image buffer at specified coords to return as the surface color.
        */
        Color value(double u, double v, const Point3& p) const override {
            if (image.height() <= 0) return Color(0, 1, 1); // cyan as debug color

            u = Interval(0, 1).clamp(u);
            v = 1.0 - Interval(0, 1).clamp(v);

            int i = int(u * image.width());
            int j = int(v * image.height());
            const unsigned char* pixel = image.pixel_data(i, j);

            double color_scale = 1.0 / 255.0;
            return Color(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
        }
};

/**
* A texture that uses the Perlin class to generate Perlin-noise like patterns.
*/
class NoiseTexture : public Texture {
    private:
        Perlin noise;
        double scale;
    public:
        /**
        * Constructs the pattern given a noise scale.
        */
        NoiseTexture(double scale) : scale(scale) {}

        /**
        * Evaluates the noise and outputs an RGB color.
        */
        Color value(double u, double v, const Point3& p) const override {
            return Color(0.5, 0.5, 0.5) * (1 + std::sin(scale * p.z() + 10 * noise.turb(p, 7)));
        }
};

#endif