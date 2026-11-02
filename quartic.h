#ifndef QUARTIC_H
#define QUARTIC_H

inline int solve_polynomial_degree_4(real a, real b, real c, real d, real e, std::vector<real>& roots) {
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

    // Guard against tiny negative discriminants from floating-point roundoff
    if (discriminant < 0.0 && discriminant > -1e-12) {
        discriminant = 0.0;
    }

    real z;

    if (discriminant >= 0.0) {
        real sd = std::sqrt(discriminant);

        real u = std::cbrt(-qb / 2.0 + sd);
        real v = std::cbrt(-qb / 2.0 - sd);

        z = u + v - rc_b / 3.0;
    } else {
        // Guard against pb slightly positive due to precision loss
        real safe_pb = std::min(real(0.0), pb);
        real sqrt_term = std::sqrt(-safe_pb / real(3.0));

        real denom = real(2.0) * sqrt_term * sqrt_term * sqrt_term;
        real ratio = (std::abs(denom) < real(1e-12)) ? real(0.0) : -qb / denom;

        real phi = std::acos(std::clamp(ratio, real(-1.0), real(1.0)));

        // Largest real root of the resolvent cubic
        z = real(2.0) * sqrt_term * std::cos(phi / real(3.0))
            - rc_b / real(3.0);
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

        virtual real evaluate_implicit(const Point3& p) const = 0;
    private:
        bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const override {
            real a, b, c, d, e;
            if (!solve_coefficients(r, a, b, c, d, e)) return false;

            std::vector<real> roots;
            if (solve_quartic(a, b, c, d, e, roots) == 0) return false;

            std::sort(roots.begin(), roots.end());

            const real min_t_epsilon = 0.001;

            for (real root : roots) {
                if (root > min_t_epsilon && ray_t.surrounds(root)) {
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

            Vec3 normal(nx, ny, nz);
            if (normal.length_squared() < 1e-12) {
                return Vec3(0.0, 0.0, 1.0);
            }
            return unit_vector(normal);
        }

        real evaluate_implicit(const Point3& p) const override {
            Point3 local_p = p - center;
            real x = local_p.x(), y = local_p.y(), z = local_p.z();
            real sum_sq = x * x + y * y + z * z;
            real mar = major_radius, mir = minor_radius;
            real term = sum_sq + mar * mar - mir * mir;
            return term * term - 4.0 * mar * mar * (x * x + z * z);
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

            real r2 = radius * radius;

            real s0 = ox * ox + oy * oy + oz * oz - 2.0 * r2;
            real s1 = 2.0 * (ox * dx + oy * dy + oz * dz);
            real s2 = dx * dx + dy * dy + dz * dz;

            real x0 = r2 - ox * ox;
            real x1 = -2.0 * ox * dx;
            real x2 = -dx * dx;

            real z0 = r2 - oz * oz;
            real z1 = -2.0 * oz * dz;
            real z2 = -dz * dz;

            a = s2 * s2 - 4.0 * x2 * z2;
            b = 2.0 * s1 * s2 - 4.0 * (x1 * z2 + x2 * z1);
            c = s1 * s1 + 2.0 * s0 * s2 - 4.0 * (x0 * z2 + x1 * z1 + x2 * z0);
            d = 2.0 * s0 * s1 - 4.0 * (x0 * z1 + x1 * z0);
            e = s0 * s0 - 4.0 * x0 * z0;

            return true;
        }

        bool clipping_check(const Point3& p, real t) const override {
            Point3 local_p = p - center;

            return std::abs(local_p.x()) <= radius &&
                std::abs(local_p.y()) <= radius * 2.0 &&
                std::abs(local_p.z()) <= radius;
        }

        Vec3 compute_normal(const Point3& p) const override {
            Point3 local_p = p - center;

            real x = local_p.x();
            real y = local_p.y();
            real z = local_p.z();

            real sum_sq = x * x + y * y + z * z;
            real r2 = radius * radius;

            real nx = 4.0 * x * (sum_sq - 2.0 * y * y);
            real ny = 4.0 * y * (sum_sq - 2.0 * x * x);
            real nz = 4.0 * z * (sum_sq - 2.0 * r2);

            Vec3 normal(nx, ny, nz);
            if (normal.length_squared() < 1e-12) {
                return Vec3(0.0, 0.0, 1.0);
            }
            return unit_vector(normal);
        }

        real evaluate_implicit(const Point3& p) const override {
            Point3 local_p = p - center;
            real x = local_p.x(), y = local_p.y(), z = local_p.z();
            real r2 = radius * radius;
            real s = x * x + y * y + z * z - 2.0 * r2;
            return s * s - 4.0 * (r2 - x * x) * (r2 - z * z);
        }

    public:
        BohemianDome(const Point3& center, real radius, shared_ptr<Material> mat) : Quartic(mat), center(center), radius(radius) {

            bbox = Aabb(
                center - Vec3(radius, radius * 2.0, radius),
                center + Vec3(radius, radius * 2.0, radius));
        }
};

class ChairSurface : public Quartic {
    private:
        Point3 center;
        real shape_a;
        real shape_b;
        real scale_k;

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

            real k = scale_k;
            real k2 = k * k;

            real q0 = ox * ox + oy * oy + oz * oz - shape_a * k2;
            real q1 = 2.0 * (ox * dx + oy * dy + oz * dz);
            real q2 = dx * dx + dy * dy + dz * dz;

            real u0 = (oz - k) * (oz - k) - 2.0 * ox * ox;
            real u1 = 2.0 * (oz - k) * dz - 4.0 * ox * dx;
            real u2 = dz * dz - 2.0 * dx * dx;

            real v0 = (oz + k) * (oz + k) - 2.0 * oy * oy;
            real v1 = 2.0 * (oz + k) * dz - 4.0 * oy * dy;
            real v2 = dz * dz - 2.0 * dy * dy;

            a = q2 * q2 - shape_b * u2 * v2;
            b = 2.0 * q1 * q2 - shape_b * (u1 * v2 + u2 * v1);
            c = q1 * q1 + 2.0 * q0 * q2 - shape_b * (u0 * v2 + u1 * v1 + u2 * v0);
            d = 2.0 * q0 * q1 - shape_b * (u0 * v1 + u1 * v0);
            e = q0 * q0 - shape_b * u0 * v0;

            return true;
        }

        // Override hit to intersect with a bounding cylinder interval first
        bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const override {
            Point3 oc = r.origin() - center;
            real ox = oc.x(), oy = oc.y();
            real dx = r.direction().x(), dy = r.direction().y();

            // 1. Intersect ray with infinite cylinder in xy-plane: x^2 + y^2 <= R^2
            real max_radius = scale_k * 0.95;
            real A_cyl = dx * dx + dy * dy;
            real B_cyl = 2.0 * (ox * dx + oy * dy);
            real C_cyl = ox * ox + oy * oy - max_radius * max_radius;

            real disc_cyl = B_cyl * B_cyl - 4.0 * A_cyl * C_cyl;
            if (disc_cyl < 0.0) return false;

            real sqrd = std::sqrt(disc_cyl);
            real t0 = (-B_cyl - sqrd) / (2.0 * A_cyl);
            real t1 = (-B_cyl + sqrd) / (2.0 * A_cyl);

            if (t0 > t1) std::swap(t0, t1);

            // Factor in height bounds (z-axis limits)
            real max_height = scale_k * 1.05;
            real cyl_tmin = std::max(ray_t.min, t0);
            real cyl_tmax = std::min(ray_t.max, t1);

            if (cyl_tmin > cyl_tmax) return false;

            // 2. Solve quartic coefficients
            real a, b, c, d, e;
            if (!solve_coefficients(r, a, b, c, d, e)) return false;

            std::vector<real> roots;
            if (solve_quartic(a, b, c, d, e, roots) == 0) return false;

            std::sort(roots.begin(), roots.end());

            const real min_t_epsilon = 0.001;

            for (real root : roots) {
                // Ensure root is within the valid cylinder interval and ray bounds
                if (root > min_t_epsilon && root >= cyl_tmin && root <= cyl_tmax && ray_t.surrounds(root)) {
                    Point3 p = r.at(root);
                    
                    // Final height check on z
                    if (std::abs(p.z() - center.z()) <= max_height) {
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

        Vec3 compute_normal(const Point3& p) const override {
            Point3 local_p = p - center;

            real x = local_p.x();
            real y = local_p.y();
            real z = local_p.z();

            real k = scale_k;
            real k2 = k * k;

            real sum_sq = x * x + y * y + z * z;
            real q = sum_sq - shape_a * k2;

            real u = (z - k) * (z - k) - 2.0 * x * x;
            real v = (z + k) * (z + k) - 2.0 * y * y;

            real nx = 4.0 * x * q + 4.0 * shape_b * x * v;
            real ny = 4.0 * y * q + 4.0 * shape_b * y * u;
            real nz = 4.0 * z * q - 2.0 * shape_b * ((z - k) * v + (z + k) * u);

            Vec3 normal(nx, ny, nz);
            if (normal.length_squared() < 1e-12) {
                return Vec3(0.0, 0.0, 1.0);
            }
            return unit_vector(normal);
        }

        real evaluate_implicit(const Point3& p) const override {
            Point3 local_p = p - center;
            real x = local_p.x(), y = local_p.y(), z = local_p.z();
            real k = scale_k;
            real q = x * x + y * y + z * z - shape_a * (k * k);
            real u = (z - k) * (z - k) - 2.0 * x * x;
            real v = (z + k) * (z + k) - 2.0 * y * y;
            return q * q - shape_b * u * v;
        }
    
    public:
        ChairSurface(const Point3& center, real shape_a, real shape_b, real scale_k, shared_ptr<Material> mat) 
            : Quartic(mat), center(center), shape_a(shape_a), shape_b(shape_b), scale_k(scale_k) {
        
            real k_abs = std::abs(scale_k);
            Vec3 extent(k_abs * 1.0, k_abs * 1.0, k_abs * 1.1);

            bbox = Aabb(center - extent, center + extent);
        }
};

#endif