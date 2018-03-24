#pragma once

#include <algorithm>

template <typename T>
struct Range {
    T left, right;

    inline T length() const {
        if (std::is_integral_v<T>)
            return right - left + 1;
        else
            return right - left;
    }

    inline bool empty() const {
        return left > right;
    }

    Range &operator+= (T value) {
        left += value;
        right += value;
        return *this;
    }

    Range operator+ (T value) const {
        return { left + value, right + value };
    }

    Range &operator-= (T value) {
        left -= value;
        right -= value;
        return *this;
    }

    Range &clampBy(const Range<T> &boundary) {
        if (left < boundary.left)
            left = boundary.left;
        if (right > boundary.right)
            right = boundary.right;
        return *this;
    }

    Range intersect(const Range<T> &rhs) const {
        return { std::max(left, rhs.left), std::min(right, rhs.right) };
    }

    template <typename Container>
    static Range<T> of(const Container &c) {
        if constexpr (std::is_same_v<decltype(c.size()), T>)
            return Range<T>{ 0, c.size() - 1 };
        else
            return Range<T>{ 0, static_cast<T>(c.size()) - 1 };
    }
};