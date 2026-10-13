#ifndef CUBE_MAP_H
#define CUBE_MAP_H

#include "common_constants.h"

enum class Face {
    pos_x,
    neg_x,
    pos_y,
    neg_y,
    pos_z,
    neg_z
};

class CubeMap {
    private:
        std::array<image, 6> faces;
    public:
        CubeMap() = default;

        CubeMap(const std::string& pos_x_file, const std::string& neg_x_file,
             const std::string& pos_y_file, const std::string& neg_y_file,
             const std::string& pos_z_file, const std::string& neg_z_file) {
                faces[static_cast<int>(face::pos_x)] = image(pos_x_file.c_str());
                faces[static_cast<int>(face::neg_x)] = image(neg_x_file.c_str());
                faces[static_cast<int>(face::pos_y)] = image(pos_y_file.c_str());
                faces[static_cast<int>(face::neg_y)] = image(neg_y_file.c_str());
                faces[static_cast<int>(face::pos_z)] = image(pos_z_file.c_str());
                faces[static_cast<int>(face::neg_z)] = image(neg_z_file.c_str());
        }
        
        Color sample(const Vec3& direction) const {
            real abs_x = std::abs(direction.x());
            real abs_y = std::abs(direction.y());
            real abs_z = std::abs(direction.z());

            Face selected_face;
            real sc, tc, ma;

            // determine the major axis and compute projected local coords
            if (abs_x >= abs_y && abs_x >= abs_z) {
                if (direction.x() > 0) {
                    selected_face = Face::pos_x;
                    sc = -direction.z();
                    tc = -direction.y();
                    ma = abs_x;
                } else {
                    selected_face = face::neg_x;
                    sc = direction.z();
                    tc = -direction.y();
                    ma = abs_x;
                }
            } else if (abs_y >= abs_x && abs_y >= abs_z) {
                if (direction.y() > 0) {
                    selected_face = face::pos_y;
                    sc = direction.x();
                    tc = direction.z();
                    ma = abs_y;
                } else {
                    selected_face = face::neg_y;
                    sc = direction.x();
                    tc = -direction.z();
                    ma = abs_y;
                }
            } else {
                if (direction.z() > 0) {
                    selected_face = face::pos_z;
                    sc = direction.x();
                    tc = -direction.y();
                    ma = abs_z;
                } else {
                    selected_face = face::neg_z;
                    sc = -direction.x();
                    tc = -direction.y();
                    ma = abs_z;
                }
            }

            // Map from [-1, 1] range to texture coords [0, 1]
            real u = (sc / ma + 1.0) * 0.5;
            real v = (tc / ma + 1.0) * 0.5;

            return faces[static_cast<int>(selected_face)].pixel_data(u, v);
        }
};

#endif