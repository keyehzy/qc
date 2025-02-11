#pragma once

#include <cmath>
#include <vector>
#include <ostream>

#include "vec3.h"
#include "factorial.h"
#include "hermite_aux.h"

class Gaussian {
public:
  constexpr Gaussian(float center, int exponent, float alpha, float slater_xi = 1.0f) :
    m_center(center), m_exponent(exponent), m_alpha(alpha * slater_xi * slater_xi) {}

  constexpr float eval(float x) const noexcept {
    float r = x - m_center;
    return std::pow(r, m_exponent) * std::exp(-m_alpha * r * r);
  }

  constexpr float overlap(const Gaussian& other) const noexcept {
    return HermiteAuxiliary::E(m_exponent, other.m_exponent, 0, other.m_center - m_center, m_alpha, other.m_alpha);
  }

  friend std::ostream& operator<<(std::ostream& os, const Gaussian& g);

private:
  float m_center;
  int m_exponent;
  float m_alpha;
};

class GaussianTypeOrbital {
public:
  struct Exponent {
    int i, j, k;
  };

  constexpr GaussianTypeOrbital(float coefficient, Vec3 center, Exponent exponent, float alpha) :
    m_coefficient(coefficient), m_center(center), m_exponent(exponent), m_alpha(alpha),
    m_norm(compute_norm(alpha, exponent.i, exponent.j, exponent.k)),
    m_A(center.x, exponent.i, alpha), m_B(center.y, exponent.j, alpha), m_C(center.z, exponent.k, alpha) {}

  constexpr float eval(Vec3 r) const noexcept {
    float total_eval = m_A.eval(r.x) * m_B.eval(r.y) * m_C.eval(r.z);
    return m_norm * m_coefficient * total_eval;
  }

  constexpr float overlap(const GaussianTypeOrbital& other) const noexcept {
    float sign = 2.0f * ((other.m_exponent.i + other.m_exponent.j + other.m_exponent.k) % 2) - 1.0f;
    float prefactor = std::pow(M_PI / (m_alpha + other.m_alpha), 1.5f);
    float total_norm = m_norm * other.m_norm;
    float total_coefficient = m_coefficient * other.m_coefficient;
    float total_overlap = m_A.overlap(other.m_A) * m_B.overlap(other.m_B) * m_C.overlap(other.m_C);
    return sign * prefactor * total_norm * total_coefficient * total_overlap;
  }

  constexpr Vec3 dipole_moment(const GaussianTypeOrbital& other) {
    Vec3 independent_term = (m_center + other.m_center) * 0.5f * overlap(other);
    float reduced_alpha = (other.m_alpha - m_alpha) / (other.m_alpha * m_alpha);
    Vec3 dependent_term{
      0.5f * ((m_exponent.i - other.m_exponent.i) * overlap(other.with_incremented_exponents(-1, 0, 0)) + 0.5f * reduced_alpha * overlap(other.with_incremented_exponents(1, 0, 0))),
      0.5f * ((m_exponent.j - other.m_exponent.j) * overlap(other.with_incremented_exponents(0, -1, 0)) + 0.5f * reduced_alpha * overlap(other.with_incremented_exponents(0, 1, 0))),
      0.5f * ((m_exponent.i - other.m_exponent.i) * overlap(other.with_incremented_exponents(0, 0, -1)) + 0.5f * reduced_alpha * overlap(other.with_incremented_exponents(0, 0, 1)))
    };
    return independent_term + dependent_term;
  }

  constexpr float kinetic(const GaussianTypeOrbital& other) const noexcept {
    return -0.5f * (overlap(other.with_incremented_exponents(2,0,0)) +
                    overlap(other.with_incremented_exponents(0,2,0)) +
                    overlap(other.with_incremented_exponents(0,0,2)));
  }

  constexpr float norm() const noexcept {
    return m_norm;
  }

  friend std::ostream& operator<<(std::ostream& os, const GaussianTypeOrbital& g);

private:
  constexpr GaussianTypeOrbital with_incremented_exponents(int i, int j, int k) const {
    GaussianTypeOrbital g = *this;
    g.m_exponent.i += i;
    g.m_exponent.j += j;
    g.m_exponent.i += k;
    return g;
  }

  static constexpr float compute_norm(float alpha, int i, int j, int k) noexcept {
    float prefactor = std::pow(2.0f * alpha / M_PI, 0.75f);
    float numerator = std::pow(4.0f * alpha, 0.5f * (i + j + k));
    float denominator = std::sqrt(double_factorial(2 * i - 1) * double_factorial(2 * j - 1) * double_factorial(2 * k - 1));
    return prefactor * numerator / denominator;
  }

  float m_coefficient;
  Vec3 m_center;
  Exponent m_exponent;
  float m_alpha;
  float m_norm;
  Gaussian m_A, m_B, m_C;
};

class ContractedGaussianTypeOrbital {
public:
  struct Param {
    float coefficient;
    float alpha;
  };

  constexpr ContractedGaussianTypeOrbital(const Vec3& origin, const GaussianTypeOrbital::Exponent& exponent, const std::vector<Param>& params) {
    for (const auto& p : params) {
       m_gtos.emplace_back(p.coefficient, origin, exponent, p.alpha);
    }
  }

  constexpr float eval(Vec3 r) const noexcept {
    float result = 0;
    for (const auto& gto : m_gtos) {
      result += gto.eval(r);
    }
    return result;
  }

  constexpr float overlap(const ContractedGaussianTypeOrbital& other) const noexcept {
    float result = 0;
    for (const auto& gto1 : m_gtos) {
      for (const auto& gto2 : other.m_gtos) {
        result += gto1.overlap(gto2);
      }
    }
    return result;
  }

  constexpr float kinect(const ContractedGaussianTypeOrbital& other) const noexcept {
    float result = 0;
    for (const auto& gto1 : m_gtos) {
      for (const auto& gto2 : other.m_gtos) {
        result += gto1.kinetic(gto2);
      }
    }
    return result;
  }

  friend std::ostream& operator<<(std::ostream& os, const ContractedGaussianTypeOrbital& g);

private:
  std::vector<GaussianTypeOrbital> m_gtos;
};

