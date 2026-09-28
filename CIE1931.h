#ifndef CIE1931_H
#define CIE1931_H

#include "common_constants.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

/**
* Dataset for spectral to RGB color conversion.
*/
class CIE1931 {
    private:
        static constexpr int MIN_WAVELENGTH = 360;
        static constexpr int MAX_WAVELENGTH = 830;
        static constexpr int NUM_SAMPLES = 471;

        real x_bar[NUM_SAMPLES]{}; // {} to init all to 0
        real y_bar[NUM_SAMPLES]{};
        real z_bar[NUM_SAMPLES]{};

    public:
        CIE1931() {
            std::ifstream file("data/CIE_xyz_1931_2deg.csv");

            if (!file) {
                throw std::runtime_error("Could not open CIE1931 dataset");
            }

            std::string line;
            while (std::getline(file, line)) {
                std::stringstream ss(line);
                int wavelength;
                std::string value;

                std::getline(ss, value, ',');
                wavelength = std::stoi(value);

                int index = wavelength - MIN_WAVELENGTH;

                std::getline(ss, value, ',');
                x_bar[index] = std::stod(value);

                std::getline(ss, value, ',');
                y_bar[index] = std::stod(value);

                std::getline(ss, value, ',');
                z_bar[index] = std::stod(value);
            }
        }

        real x(int wavelength) const {
            return x_bar[wavelength - MIN_WAVELENGTH];
        }

        real y(int wavelength) const {
            return y_bar[wavelength - MIN_WAVELENGTH];
        }

        real z(int wavelength) const {
            return z_bar[wavelength - MIN_WAVELENGTH];
        }
};

#endif