#ifndef TEXTURE_H
#define TEXTURE_H

#include "perlin.h"
#include "rtw_stb_image.h"

class Texture {
    public:
        virtual ~Texture() = default;

        virtual Color value(double u, double v, const Point3& p) const = 0;
};

class SolidColor : public Texture {
    private:
        Color albedo;
    public:
        SolidColor(const Color& albedo) : albedo(albedo) {}
        SolidColor(double red, double green, double blue) : SolidColor(Color(red, green, blue)) {}

        Color value(double u, double v, const Point3& p) const override { return albedo; }
};

class CheckerTexture : public Texture {
    private:
        double inv_scale;
        shared_ptr<Texture> even;
        shared_ptr<Texture> odd;
    public:
        CheckerTexture(double scale, shared_ptr<Texture> even, shared_ptr<Texture> odd)
        : inv_scale(1.0 / scale), even(even), odd(odd) {}

        CheckerTexture(double scale, const Color& c1, const Color& c2) 
        : CheckerTexture(scale, make_shared<SolidColor>(c1), make_shared<SolidColor>(c2)) {}

        Color value(double u, double v, const Point3& p) const override {
            int xInt = int(std::floor(inv_scale * p.x()));
            int yInt = int(std::floor(inv_scale * p.y()));
            int zInt = int(std::floor(inv_scale * p.z()));

            bool isEven = (xInt + yInt + zInt) % 2 == 0;

            return isEven ? even->value(u, v, p) : odd->value(u, v, p);
        }
};

class ImageTexture : public Texture {
    private:
        RtwImage image;
    public:
        ImageTexture(const char* filename) : image(filename) {}

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

class NoiseTexture : public Texture {
    private:
        Perlin noise;
        double scale;
    public:
        NoiseTexture(double scale) : scale(scale) {}

        Color value(double u, double v, const Point3& p) const override {
            return Color(0.5, 0.5, 0.5) * (1 + std::sin(scale * p.z() + 10 * noise.turb(p, 7)));
        }
};

#endif