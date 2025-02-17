#pragma once

#include <cmath>

struct Vec3 {
  float x, y, z;

  constexpr Vec3 operator+(const Vec3& other) const {
    return Vec3{x + other.x, y + other.y, z + other.z};
  }

  constexpr Vec3 operator-(const Vec3& other) const {
    return Vec3{x - other.x, y - other.y, z - other.z};
  }

  constexpr Vec3 operator*(float k) const {
    return Vec3{x * k, y * k, z * k};
  }

  constexpr Vec3 operator/(float k) const {
    return Vec3{x / k, y / k, z / k};
  }

  constexpr float norm() const {
    return std::sqrt(x*x+y*y+z*z);
  }

  friend constexpr Vec3 operator*(float a, const Vec3& b) {
    return b * a;
  }
};
