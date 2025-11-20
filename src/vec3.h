#pragma once

#include <cmath>

struct Vec3 {
  double x, y, z;

  constexpr Vec3 operator+(const Vec3& other) const {
    return Vec3{x + other.x, y + other.y, z + other.z};
  }

  constexpr Vec3 operator-(const Vec3& other) const {
    return Vec3{x - other.x, y - other.y, z - other.z};
  }

  constexpr Vec3 operator*(double k) const {
    return Vec3{x * k, y * k, z * k};
  }

  constexpr Vec3 operator/(double k) const {
    return Vec3{x / k, y / k, z / k};
  }

  constexpr double norm() const {
    return std::sqrt(x*x+y*y+z*z);
  }

  constexpr double norm2() const {
    return x*x+y*y+z*z;
  }

  friend constexpr Vec3 operator*(double a, const Vec3& b) {
    return b * a;
  }
};
