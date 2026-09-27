#ifndef INTERVAL_H
#define INTERVAL_H

/**
* Manages standard intervals.
*/
class Interval {
    public:
        real min;
        real max;

        static const Interval empty;
        static const Interval universe;

        Interval() : min(+infinity), max(-infinity) {}

        Interval(real min, real max) : min(min), max(max) {}

        real size() const {
            return max - min;
        }

        bool contains(real x) const {
            return min <= x && x <= max;
        }

        bool surrounds(real x) const {
            return min < x && x < max;
        }

        real clamp(real x) const {
            if (x < min) return min;
            if (x > max) return max;
            return x;
        }

        /**
        * Used for Aabb bounding boxes to expand interval by delta / 2
        */
        Interval expand(real delta) const {
            real padding = delta / 2;
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
Interval operator+(const Interval& ival, real displacement) {
    return Interval(ival.min + displacement, ival.max + displacement);
}

Interval operator+(real displacement, const Interval& ival) {
    return ival + displacement;
}

#endif