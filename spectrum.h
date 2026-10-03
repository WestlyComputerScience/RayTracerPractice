#ifndef SPECTRUM_H
#define SPECTRUM_H

#include "common_constants.h"
#include "xyz.h"
#include "CIE1931.h"

static constexpr real MIN_WAVELENGTH = 360;
static constexpr real MAX_WAVELENGTH = 830;

#ifdef SPECTRAL_20NM
    static constexpr int WAVELENGTH_STEP = 20; // 24 samples
#else
    static constexpr int WAVELENGTH_STEP = 1; // 471 samples
#endif

static constexpr int NUM_SAMPLES = (MAX_WAVELENGTH - MIN_WAVELENGTH) / WAVELENGTH_STEP + 1;

/**
* This class represents the implementation of spectral colors as opposed to RGB colors while still using a CIE converter method
* to change spectral colors to RGB colors for the monitor to read.
* 
* The spectral values in CIE 1931 are inspired by https://registry.color.org/colorimetry-data, I used the
* "CIE 1931 colour matching functions, 2 degree observer" dataset.
*/
class Spectrum {
    private:
        real spectralValues[NUM_SAMPLES];
        static constexpr real CIE_Y_INTEGRAL = 106.8568;
    public:
    /**
    * Creates a spectrum with all wavelengths at zero.
    */
    Spectrum() {
        for (int i = 0; i < NUM_SAMPLES; i++) {
            spectralValues[i] = 0;
        }
    }

    /**
    * Creates a spectrum where every wavelength has the same value.
    */
    Spectrum(real spectralValue) {
        for (int i = 0; i < NUM_SAMPLES; i++) {
            spectralValues[i] = spectralValue;
        }
    }

    /**
    * return the wavelength (nm) from the spectral values.
    */
    static real wavelength(int index) {
        return MIN_WAVELENGTH + index * WAVELENGTH_STEP;
    }

    /**
    * Gets the value of a wavelength sample.
    */
    real operator[](int index) const {
        return spectralValues[index];
    }

    real& operator[](int index) {
        return spectralValues[index];
    }

    /**
    * Spectrum arithmetic.
    */
    Spectrum operator+(const Spectrum& s) const {
        Spectrum result;
        for (int i = 0; i < NUM_SAMPLES; i++) {
            result[i] = spectralValues[i] + s[i];
        }
        return result;
    }

    Spectrum operator*(const Spectrum& s) const {
        Spectrum result;
        for (int i = 0; i < NUM_SAMPLES; i++) {
            result[i] = spectralValues[i] * s[i];
        }
        return result;
    }

    Spectrum operator*(real t) const {
        Spectrum result;
        for (int i = 0; i < NUM_SAMPLES; i++) {
            result[i] = spectralValues[i] * t;
        }
        return result;
    }

    Spectrum& operator+=(const Spectrum& s) {
        for (int i = 0; i < NUM_SAMPLES; i++) {
            spectralValues[i] += s[i];
        }
        return *this;
    }

    XYZ toXYZ(const CIE1931& cie) const {
        XYZ result;
        result.x = 0;
        result.y = 0;
        result.z = 0;

        for (int i = 0; i < NUM_SAMPLES; i++) {
            int lambda = wavelength(i);
            
            result.x += spectralValues[i] * cie.x(lambda) * WAVELENGTH_STEP;
            result.y += spectralValues[i] * cie.y(lambda) * WAVELENGTH_STEP;
            result.z += spectralValues[i] * cie.z(lambda) * WAVELENGTH_STEP;
        }
        result.x /= CIE_Y_INTEGRAL;
        result.y /= CIE_Y_INTEGRAL;
        result.z /= CIE_Y_INTEGRAL;

        return result;
    }
};

/**
* Mulitplies every wavelength by a scalar.
*/
inline Spectrum operator*(real t, const Spectrum& s) {
    return s * t;
}

#endif