#ifndef RTW_STB_IMAGE_H
#define RTW_STB_IMAGE_H

// disable strict warnings for this header from compiler
#ifdef _MSC_VER
    #pragma warning (push, 0)
#endif

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "external/stb_image.h"

#include <cstdlib>
#include <iostream>

class RtwImage {
    private:
        const int bytes_per_pixel = 3;
        float *fdata = nullptr;
        unsigned char *bdata = nullptr;
        int image_width = 0;
        int image_height = 0;
        int bytes_per_scanline = 0;

        static int clamp(int x, int low, int high) {
            if (x < low) return low;
            if (x < high) return x;
            return high - 1;
        }

        static unsigned char float_to_byte(float value) {
            if (value <= 0.0) return 0;
            if (1.0 <= value) return 255;
            return static_cast<unsigned char>(256.0 * value);
        }

        void convert_to_bytes() {
            int total_bytes = image_width * image_height * bytes_per_pixel;
            bdata = new unsigned char[total_bytes];

            unsigned char *bptr = bdata;
            float *fptr = fdata;
            for (int i = 0; i < total_bytes; i++, fptr++, bptr++) {
                *bptr = float_to_byte(*fptr);
            }
        }
    public:
        RtwImage() {}

        /**
        * Loads image data from specified file, if the RTW_IMAGES environment variable is defined, it looks only in that directory for the image file.
        * If image wasn't found, searches for it from the current directory, then in the images/subdirectory, then the parent's images/subdirectory, 
        * and that parent etc. for 3 levels up. If image wasn't loaded successfully, width() and height() will return 0.
        */
        RtwImage(const char* image_filename) {
            std::string filename = std::string(image_filename);
            const char* imagedir = getenv("RTW_IMAGES");

            if (imagedir && load(std::string(imagedir) + "/" + image_filename)) return;
            if (load(filename)) return;
            if (load("images/" + filename)) return;
            if (load("../images/" + filename)) return;
            if (load("../../images/" + filename)) return;
            if (load("../../../images/" + filename)) return;

            std::cerr << "ERROR: Could not load image file " << image_filename << "\n";
        }

        ~RtwImage() {
            delete[] bdata;
            STBI_FREE(fdata);
        }

        bool load(const std::string& filename) {
            int n = bytes_per_pixel;
            fdata = stbi_loadf(filename.c_str(), &image_width, &image_height, &n, bytes_per_pixel);
            if (fdata == nullptr) return false;

            bytes_per_scanline = image_width * bytes_per_pixel;
            convert_to_bytes();
            return true;
        }

        int width() const { return (fdata == nullptr) ? 0 : image_width; }
        int height() const { return (fdata == nullptr) ? 0 : image_height; }

        const unsigned char* pixel_data(int x, int y) const {
            static unsigned char magenta[] = { 255, 0, 255 };
            if (bdata == nullptr) return magenta;

            x = clamp(x, 0, image_width);
            y = clamp(y, 0, image_height);

            return bdata + y*bytes_per_scanline + x*bytes_per_pixel;
        }
};

// restore the compiler warnings
#ifdef _MSC_VER
    #pragma warning (pop)
#endif

#endif