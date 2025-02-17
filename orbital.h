#pragma once

#include <cmath>
#include <vector>
#include <ostream>

#include "vec3.h"
#include "factorial.h"
#include "hermite_aux.h"

class Gaussian {
public:
  constexpr Gaussian(float center, int exponent, float alpha) :
    m_center(center), m_exponent(exponent), m_alpha(alpha) {}

  constexpr float eval(float x) const noexcept {
    float r = x - m_center;
    return std::pow(r, m_exponent) * std::exp(-m_alpha * r * r);
  }

  constexpr float overlap(const Gaussian& other, int nodes = 0) const noexcept {
    return HermiteAuxiliary::hermite_E(m_exponent, other.m_exponent, nodes, m_center - other.m_center, m_alpha, other.m_alpha);
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

  constexpr GaussianTypeOrbital(float coefficient, const Vec3& center, Exponent exponent, float alpha) :
    m_coefficient(coefficient), m_center(center), m_exponent(exponent), m_alpha(alpha),
    m_norm(compute_norm(alpha, exponent.i, exponent.j, exponent.k)),
    m_A(center.x, exponent.i, alpha), m_B(center.y, exponent.j, alpha), m_C(center.z, exponent.k, alpha) {}

  constexpr float eval(Vec3 r) const noexcept {
    float total_eval = m_A.eval(r.x) * m_B.eval(r.y) * m_C.eval(r.z);
    return m_norm * m_coefficient * total_eval;
  }

  constexpr float overlap(const GaussianTypeOrbital& other) const noexcept {
    float total_norm = m_norm * other.m_norm;
    float total_coefficient = m_coefficient * other.m_coefficient;
    return total_norm * total_coefficient * overlap(other);
  }

  constexpr float kinetic(const GaussianTypeOrbital& other) const noexcept {
    float term_A = other.m_alpha * (2.0f * (other.m_exponent.i + other.m_exponent.j + other.m_exponent.k) + 3.0f) * unnormalized_overlap(other);
    float term_B = -2.0f * other.m_alpha * other.m_alpha * (unnormalized_overlap(other.with_incremented_exponents(2,0,0)) +
                                                            unnormalized_overlap(other.with_incremented_exponents(0,2,0)) +
                                                            unnormalized_overlap(other.with_incremented_exponents(0,0,2)));
    float term_C = -0.5f * (other.m_exponent.i * (other.m_exponent.i - 1) * unnormalized_overlap(other.with_incremented_exponents(-2,0,0)) +
                            other.m_exponent.j * (other.m_exponent.j - 1) * unnormalized_overlap(other.with_incremented_exponents(0,-2,0)) +
                            other.m_exponent.k * (other.m_exponent.k - 1) * unnormalized_overlap(other.with_incremented_exponents(0,0,-2)));
    float total_norm = m_norm * other.m_norm;
    float total_coefficient = m_coefficient * other.m_coefficient;
    float total_kinetic = term_A + term_B + term_C;
    return total_norm * total_coefficient * total_kinetic;
  }

  constexpr float nuclear(const GaussianTypeOrbital& other, const Vec3& nucleus_center) const noexcept {
    float combined_alpha = m_alpha + other.m_alpha;
    Vec3 composite_center = (m_alpha * m_center + other.m_alpha * other.m_center) / combined_alpha;
    float result = 0.0f;
    for (int i = 0; i < m_exponent.i + other.m_exponent.i + 1; i++) {
      for (int j = 0; j < m_exponent.j + other.m_exponent.j + 1; j++) {
        for (int k = 0; k < m_exponent.k + other.m_exponent.k + 1; k++) {
          result += m_A.overlap(other.m_A, i) * m_B.overlap(other.m_B, j) * m_C.overlap(other.m_C, k) *
            HermiteAuxiliary::hermite_R(i, j, k, 0, combined_alpha, composite_center - nucleus_center);
        }
      }
    }
    float prefactor = 2.0f * M_PI / combined_alpha;
    float total_norm = m_norm * other.m_norm;
    float total_coefficient = m_coefficient * other.m_coefficient;
    return prefactor * total_norm * total_coefficient * result;
  }

  constexpr float norm() const noexcept {
    return m_norm;
  }

  constexpr float coefficient() const noexcept {
    return m_coefficient;
  }

  friend std::ostream& operator<<(std::ostream& os, const GaussianTypeOrbital& g);

private:
  constexpr float unnormalized_overlap(const GaussianTypeOrbital& other) const noexcept {
    float prefactor = std::pow(M_PI / (m_alpha + other.m_alpha), 1.5f);
    float total_overlap = m_A.overlap(other.m_A) * m_B.overlap(other.m_B) * m_C.overlap(other.m_C);
    return prefactor * total_overlap;
  }

  constexpr GaussianTypeOrbital with_incremented_exponents(int di, int dj, int dk) const noexcept {
    Exponent new_exponent{ m_exponent.i + di, m_exponent.j + dj, m_exponent.k + dk };
    return GaussianTypeOrbital(m_coefficient, m_center, new_exponent, m_alpha);
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
      m_gtos.push_back({p.coefficient, origin, exponent, p.alpha});
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

  constexpr float kinetic(const ContractedGaussianTypeOrbital& other) const noexcept {
    float result = 0;
    for (const auto& gto1 : m_gtos) {
      for (const auto& gto2 : other.m_gtos) {
        result += gto1.kinetic(gto2);
      }
    }
    return result;
  }

  constexpr float nuclear(const ContractedGaussianTypeOrbital& other, const Vec3& nucleus_center) const noexcept {
    float result = 0;
    for (const auto& gto1 : m_gtos) {
      for (const auto& gto2 : other.m_gtos) {
        result += gto1.nuclear(gto2, nucleus_center);
      }
    }
    return result;
  }

  friend std::ostream& operator<<(std::ostream& os, const ContractedGaussianTypeOrbital& g);

private:
  std::vector<GaussianTypeOrbital> m_gtos;
};

