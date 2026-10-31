#ifndef QUARTIC_H
#define QUARTIC_H


inline int solve_polynomial_degree_4(
    real a, real b, real c, real d, real e,
    std::vector<real>& roots
) {
    if (std::abs(a) < 1e-12) return 0;

    // Normalize coefficients
    b /= a;
    c /= a;
    d /= a;
    e /= a;

    // Convert to depressed quartic:
    // y^4 + p*y^2 + q*y + r = 0
    real b2 = b * b;

    real p = c - 0.375 * b2;
    real q = d - 0.5 * b * c + 0.125 * b2 * b;
    real r = e - 0.25 * b * d
               + 0.0625 * b2 * c
               - 0.01171875 * b2 * b2;

    // Special case: biquadratic equation (q = 0)
    if (std::abs(q) < 1e-12) {
        real disc = p * p - 4.0 * r;

        if (disc >= 0.0) {
            real sq_disc = std::sqrt(disc);

            real y1 = (-p - sq_disc) * 0.5;
            real y2 = (-p + sq_disc) * 0.5;

            if (y1 >= 0.0) {
                real sy = std::sqrt(y1);

                roots.push_back(sy - 0.25 * b);
                roots.push_back(-sy - 0.25 * b);
            }

            if (y2 >= 0.0 && y2 != y1) {
                real sy = std::sqrt(y2);

                roots.push_back(sy - 0.25 * b);
                roots.push_back(-sy - 0.25 * b);
            }
        }

        return static_cast<int>(roots.size());
    }

    // Solve Ferrari's resolvent cubic:
    // z^3 + 2p*z^2 + (p^2 - 4r)*z - q^2 = 0

    real rc_b = 2.0 * p;
    real rc_c = p * p - 4.0 * r;
    real rc_d = -q * q;

    // Convert cubic to depressed cubic
    real pb = rc_c - rc_b * rc_b / 3.0;

    real qb = (
        2.0 * rc_b * rc_b * rc_b
        - 9.0 * rc_b * rc_c
        + 27.0 * rc_d
    ) / 27.0;

    real discriminant =
        qb * qb / 4.0 + pb * pb * pb / 27.0;

    real z;

    if (discriminant >= 0.0) {
        real sd = std::sqrt(discriminant);

        real u = std::cbrt(-qb / 2.0 + sd);
        real v = std::cbrt(-qb / 2.0 - sd);

        z = u + v - rc_b / 3.0;
    }
    else {
        real sqrt_term = std::sqrt(-pb / 3.0);

        real phi = std::acos(
            std::clamp(
                -qb / (2.0 * sqrt_term * sqrt_term * sqrt_term),
                -1.0, 1.0
            )
        );

        // Largest real root of the resolvent cubic
        z = 2.0 * sqrt_term * std::cos(phi / 3.0)
            - rc_b / 3.0;
    }

    // z = m^2
    if (z <= 0.0) return 0;

    real m = std::sqrt(z);

    // Factor into two quadratic equations:
    //
    // (y^2 + m*y + A)(y^2 - m*y + B)
    //
    // A = (p + z - q/m) / 2
    // B = (p + z + q/m) / 2

    real A = (p + z - q / m) / 2.0;
    real B = (p + z + q / m) / 2.0;

    // Solve first quadratic: y^2 + m*y + A
    real d1 = m * m - 4.0 * A;

    if (d1 >= 0.0) {
        real sd1 = std::sqrt(d1);

        roots.push_back((-m - sd1) * 0.5 - 0.25 * b);
        roots.push_back((-m + sd1) * 0.5 - 0.25 * b);
    }

    // Solve second quadratic: y^2 - m*y + B
    real d2 = m * m - 4.0 * B;

    if (d2 >= 0.0) {
        real sd2 = std::sqrt(d2);

        roots.push_back((m - sd2) * 0.5 - 0.25 * b);
        roots.push_back((m + sd2) * 0.5 - 0.25 * b);
    }

    return static_cast<int>(roots.size());
}

class Quartic : public Hittable {
    private:
        virtual int solve_quartic(real a, real b, real c, real d, real e, std::vector<real>& roots) const = 0;
    protected:
        Aabb bbox;
        shared_ptr<Material> mat;

        Quartic(shared_ptr<Material> mat) : mat(mat) {}

        virtual bool solve_coefficients(const Ray& r, real& a, real& b, real& c, real& d, real& e) const = 0;

        virtual bool clipping_check(const Point3& p, real t) const { return true; }

        virtual Vec3 compute_normal(const Point3& p) const = 0;
    private:
        bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const override {
            real a, b, c, d, e;
            if (!solve_coefficients(r, a, b, c, d, e)) return false;

            std::vector<real> roots;
            if (solve_quartic(a, b, c, d, e, roots) == 0) return false;

            std::sort(roots.begin(), roots.end());

            for (real root : roots) {
                if (ray_t.surrounds(root)) {
                    Point3 p = r.at(root);
                    if (clipping_check(p, root)) {
                        rec.t = root;
                        rec.p = p;
                        Vec3 outward_normal = compute_normal(p);
                        rec.set_face_normal(r, outward_normal);
                        rec.mat = mat;
                        return true;
                    }
                }
            }

            return false;
        }

        Aabb bounding_box() const override { return bbox; }
};

class Torus : public Quartic {
    private:
        Point3 center;
        real major_radius;
        real minor_radius;

        int solve_quartic(real a, real b, real c, real d, real e, std::vector<real>& roots) const override {
            return solve_polynomial_degree_4(a, b, c, d, e, roots);
        }

        bool solve_coefficients(const Ray& r, real& a, real& b, real& c, real& d, real& e) const override {
            Point3 oc = r.origin() - center;

            real mar = major_radius;
            real mir = minor_radius;

            real dx = r.direction().x();
            real dy = r.direction().y();
            real dz = r.direction().z();

            real ox = oc.x();
            real oy = oc.y();
            real oz = oc.z();

            real sum_d_sq = dx * dx + dy * dy + dz * dz;
            real oc_dot_d = ox * dx + oy * dy + oz * dz;
            real oc_sq = ox * ox + oy * oy + oz * oz;

            real sum_d_xz_sq = dx * dx + dz * dz;
            real oc_dot_d_xz = ox * dx + oz * dz;
            real oc_xz_sq = ox * ox + oz * oz;

            real m = oc_sq + mar * mar - mir * mir;

            a = sum_d_sq * sum_d_sq;
            b = 4.0 * sum_d_sq * oc_dot_d;
            c = 2.0 * sum_d_sq * m + 4.0 * (oc_dot_d * oc_dot_d) - 4.0 * mar * mar * sum_d_xz_sq;
            d = 4.0 * oc_dot_d * m - 8.0 * mar * mar * oc_dot_d_xz;
            e = m * m - 4.0 * mar * mar * oc_xz_sq;

            return true;
        }
        
        Vec3 compute_normal(const Point3& p) const override {
            Point3 local_p = p - center;

            real x = local_p.x();
            real y = local_p.y();
            real z = local_p.z();

            real sum_sq = x * x + y * y + z * z;
            real mar = major_radius;
            real mir = minor_radius;

            real common = 4.0 * (sum_sq + mar * mar - mir * mir);

            real nx = (common - 8.0 * mar * mar) * x;
            real ny = common * y;
            real nz = (common - 8.0 * mar * mar) * z;

            return unit_vector(Vec3(nx, ny, nz));
        }

    public:
        Torus(const Point3& center, real major_radius, real minor_radius, shared_ptr<Material> mat)
            : Quartic(mat), center(center), major_radius(major_radius), minor_radius(minor_radius) {
                real total_radius = major_radius + minor_radius;
                bbox = Aabb(
                    center - Vec3(total_radius, minor_radius, total_radius),
                    center + Vec3(total_radius, minor_radius, total_radius)
                );
            }
};

class BohemianDome : public Quartic {
    private:
        Point3 center;
        real radius;
        real max_extent;

        int solve_quartic(real a, real b, real c, real d, real e, std::vector<real>& roots) const override {
            return solve_polynomial_degree_4(a, b, c, d, e, roots);
        }

        bool solve_coefficients(const Ray& r, real& a, real& b, real& c, real& d, real& e) const override {
            Point3 oc = r.origin() - center;

            real ox = oc.x();
            real oy = oc.y();
            real oz = oc.z();

            real dx = r.direction().x();
            real dy = r.direction().y();
            real dz = r.direction().z();

            real as = dx * dx + dy * dy - dz * dz;
            real bs = 2.0 * (ox * dx + oy * dy - oz * dz);
            real cs = ox * ox + oy * oy - oz * oz;

            real r2 = radius * radius;

            a = as * as;
            b = 2.0 * as * bs;
            c = bs * bs + 2.0 * as * cs - 4.0 * r2 * (dx * dx + dy * dy);
            d = 2.0 * bs * cs - 8.0 * r2 * (ox * dx + oy * dy);
            e = cs * cs - 4.0 * r2 * (ox * ox + oy * oy);

            return true;
        }

        bool clipping_check(const Point3& p, real t) const override {
            Point3 local_p = p - center;

            real x = local_p.x();
            real y = local_p.y();
            real z = local_p.z();

            real xy_extent = radius * 2.0;
            real z_extent = radius * 2.0 * std::sqrt(2.0);

            if (std::abs(x) > xy_extent ||
                std::abs(y) > xy_extent ||
                std::abs(z) > z_extent)
                return false;

            return true;
        }

        Vec3 compute_normal(const Point3& p) const override {
            Point3 local_p = p - center;

            real x = local_p.x();
            real y = local_p.y();
            real z = local_p.z();

            real x2_y2_z2 = x * x + y * y - z * z;
            real r2 = radius * radius;

            real nx = 4.0 * x * (x2_y2_z2 - 2.0 * r2);
            real ny = 4.0 * y * (x2_y2_z2 - 2.0 * r2);
            real nz = -4.0 * z * x2_y2_z2;

            return unit_vector(Vec3(nx, ny, nz));
        }
    public:
        BohemianDome(const Point3& center, real radius, shared_ptr<Material> mat) : Quartic(mat), center(center), radius(radius), max_extent(radius * 2.0) {
            real xy_bound = radius * 2.0;
            real z_bound = radius * 2.0 * std::sqrt(2.0);

            bbox = Aabb(
                center - Vec3(xy_bound, xy_bound, z_bound),
                center + Vec3(xy_bound, xy_bound, z_bound));
        }
};

#endif