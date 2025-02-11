#pragma once

struct Vec3 {
  float x, y, z;

  constexpr Vec3 operator+(const Vec3& other) const {
    return Vec3{x + other.x, y + other.y, z + other.z};
  }

  constexpr Vec3 operator*(float k) const {
    return Vec3{k * x, k * y, k * z};
  }
};
