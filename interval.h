#ifndef INTERVAL_H
#define INTERVAL_H

/**
* Manages standard intervals.
*/
class Interval {
    public:
        double min;
        double max;

        static const Interval empty;
        static const Interval universe;

        Interval() : min(+infinity), max(-infinity) {}

        Interval(double min, double max) : min(min), max(max) {}

        double size() const {
            return max - min;
        }

        bool contains(double x) const {
            return min <= x && x <= max;
        }

        bool surrounds(double x) const {
            return min < x && x < max;
        }

        double clamp(double x) const {
            if (x < min) return min;
            if (x > max) return max;
            return x;
        }

        /**
        * Used for Aabb bounding boxes to expand interval by delta / 2
        */
        Interval expand(double delta) const {
            double padding = delta / 2;
            return Interval(min - padding, max + padding);
        }

        Interval(const Interval& a, const Interval& b) {
            min = a.min <= b.min ? a.min : b.min;
            max = a.max >= b.max ? a.max : b.max;
        }
};

const Interval Interval::empty = Interval(+infinity, -infinity);
const Interval Interval::universe = Interval(-infinity, +infinity);

/**
* Shifts the entire range by a set amount
*/
Interval operator+(const Interval& ival, double displacement) {
    return Interval(ival.min + displacement, ival.max + displacement);
}

Interval operator+(double displacement, const Interval& ival) {
    return ival + displacement;
}

#endif