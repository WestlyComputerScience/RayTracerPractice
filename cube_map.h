#ifndef CUBE_MAP_H
#define CUBE_MAP_H

#include "common_constants.h"
#include "rtw_stb_image.h"

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
        std::array<std::shared_ptr<RtwImage>, 6> faces;
    public:
        CubeMap() = default;

        CubeMap(const std::string& pos_x_file, const std::string& neg_x_file,
             const std::string& pos_y_file, const std::string& neg_y_file,
             const std::string& pos_z_file, const std::string& neg_z_file) {
            faces[static_cast<int>(Face::pos_x)] = std::make_shared<RtwImage>(pos_x_file.c_str());
            faces[static_cast<int>(Face::neg_x)] = std::make_shared<RtwImage>(neg_x_file.c_str());
            faces[static_cast<int>(Face::pos_y)] = std::make_shared<RtwImage>(pos_y_file.c_str());
            faces[static_cast<int>(Face::neg_y)] = std::make_shared<RtwImage>(neg_y_file.c_str());
            faces[static_cast<int>(Face::pos_z)] = std::make_shared<RtwImage>(pos_z_file.c_str());
            faces[static_cast<int>(Face::neg_z)] = std::make_shared<RtwImage>(neg_z_file.c_str());
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
                    selected_face = Face::neg_x;
                    sc = direction.z();
                    tc = -direction.y();
                    ma = abs_x;
                }
            } else if (abs_y >= abs_x && abs_y >= abs_z) {
                if (direction.y() > 0) {
                    selected_face = Face::pos_y;
                    sc = direction.x();
                    tc = direction.z();
                    ma = abs_y;
                } else {
                    selected_face = Face::neg_y;
                    sc = direction.x();
                    tc = -direction.z();
                    ma = abs_y;
                }
            } else {
                if (direction.z() > 0) {
                    selected_face = Face::pos_z;
                    sc = direction.x();
                    tc = -direction.y();
                    ma = abs_z;
                } else {
                    selected_face = Face::neg_z;
                    sc = -direction.x();
                    tc = -direction.y();
                    ma = abs_z;
                }
            }

            // Map from [-1, 1] range to texture coords [0, 1]
            real u = (sc / ma + 1.0) * 0.5;
            real v = (tc / ma + 1.0) * 0.5;

            const RtwImage& face_image = *faces[static_cast<int>(selected_face)];

            int width = face_image.width();
            int height = face_image.height();

            if (width == 0 || height == 0) return Color(0, 0, 0);

            // Convert [0, 1] UV into pixel int coords
            int i = static_cast<int>(u * width);
            int j = static_cast<int>(v * height);

            // Fetch pixel byte data
            const unsigned char* pixel = face_image.pixel_data(i, j);

            return Color(pixel[0] / 255.0, pixel[1] / 255.0, pixel[2] / 255.0);
        }
};

#endif