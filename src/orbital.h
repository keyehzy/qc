#pragma once

#include <cmath>
#include <vector>
#include <ostream>

#include "vec3.h"
#include "factorial.h"
#include "hermite_aux.h"

class Gaussian {
public:
  constexpr Gaussian(double center, int exponent, double alpha) :
    m_center(center), m_exponent(exponent), m_alpha(alpha) {}

  constexpr double eval(double x) const noexcept {
    double r = x - m_center;
    return std::pow(r, m_exponent) * std::exp(-m_alpha * r * r);
  }

  constexpr double overlap(const Gaussian& other, int nodes = 0) const noexcept {
    return HermiteAuxiliary::hermite_E(m_exponent, other.m_exponent, nodes, m_center - other.m_center, m_alpha, other.m_alpha);
  }

  friend std::ostream& operator<<(std::ostream& os, const Gaussian& g);

private:
  double m_center;
  int m_exponent;
  double m_alpha;
};

class GaussianTypeOrbital {
public:
  struct Exponent {
    int i, j, k;
  };

  constexpr GaussianTypeOrbital(double coefficient, const Vec3& center, Exponent exponent, double alpha) :
    m_coefficient(coefficient), m_center(center), m_exponent(exponent), m_alpha(alpha),
    m_norm(compute_norm(alpha, exponent.i, exponent.j, exponent.k)),
    m_A(center.x, exponent.i, alpha), m_B(center.y, exponent.j, alpha), m_C(center.z, exponent.k, alpha) {}

  constexpr double eval(Vec3 r) const noexcept {
    double total_eval = m_A.eval(r.x) * m_B.eval(r.y) * m_C.eval(r.z);
    return m_norm * m_coefficient * total_eval;
  }

  constexpr double overlap(const GaussianTypeOrbital& other) const noexcept {
    double total_norm = m_norm * other.m_norm;
    double total_coefficient = m_coefficient * other.m_coefficient;
    return total_norm * total_coefficient * unnormalized_overlap(other);
  }

  constexpr double kinetic(const GaussianTypeOrbital& other) const noexcept {
    double term_A = other.m_alpha *
      (2.0 * (other.m_exponent.i + other.m_exponent.j + other.m_exponent.k) + 3.0) * unnormalized_overlap(other);
    double term_B = -2.0 * other.m_alpha * other.m_alpha *
      (unnormalized_overlap(other.with_incremented_exponents(2,0,0)) +
       unnormalized_overlap(other.with_incremented_exponents(0,2,0)) +
       unnormalized_overlap(other.with_incremented_exponents(0,0,2)));
    double term_C = -0.5 *
      (other.m_exponent.i * (other.m_exponent.i - 1) * unnormalized_overlap(other.with_incremented_exponents(-2,0,0)) +
       other.m_exponent.j * (other.m_exponent.j - 1) * unnormalized_overlap(other.with_incremented_exponents(0,-2,0)) +
       other.m_exponent.k * (other.m_exponent.k - 1) * unnormalized_overlap(other.with_incremented_exponents(0,0,-2)));
    double total_norm = m_norm * other.m_norm;
    double total_coefficient = m_coefficient * other.m_coefficient;
    double total_kinetic = term_A + term_B + term_C;
    return total_norm * total_coefficient * total_kinetic;
  }

  constexpr double nuclear_attraction(const GaussianTypeOrbital& other, const Vec3& nucleus_center) const noexcept {
    double combined_alpha = m_alpha + other.m_alpha;
    Vec3 composite_center = (m_alpha * m_center + other.m_alpha * other.m_center) / combined_alpha;
    double result = 0.0;
    for (int i = 0; i < m_exponent.i + other.m_exponent.i + 1; i++) {
      for (int j = 0; j < m_exponent.j + other.m_exponent.j + 1; j++) {
        for (int k = 0; k < m_exponent.k + other.m_exponent.k + 1; k++) {
          result +=
            m_A.overlap(other.m_A, i) *
            m_B.overlap(other.m_B, j) *
            m_C.overlap(other.m_C, k) *
            HermiteAuxiliary::hermite_R(i, j, k, 0, combined_alpha, composite_center - nucleus_center);
        }
      }
    }
    double prefactor = 2.0 * M_PI / combined_alpha;
    double total_norm = m_norm * other.m_norm;
    double total_coefficient = m_coefficient * other.m_coefficient;
    return prefactor * total_norm * total_coefficient * result;
  }

  static constexpr double electron_repulsion(const GaussianTypeOrbital& gto1, const GaussianTypeOrbital& gto2,
                                     const GaussianTypeOrbital& gto3, const GaussianTypeOrbital& gto4) noexcept {
    const double p = gto1.alpha() + gto2.alpha();
    const double q = gto3.alpha() + gto4.alpha();
    const double alpha = p * q / (p + q);

    Vec3 P = (gto1.m_alpha * gto1.m_center + gto2.m_alpha * gto2.m_center) / p;
    Vec3 Q = (gto3.m_alpha * gto3.m_center + gto4.m_alpha * gto4.m_center) / q;

    double result = 0.0;

    for (int t = 0; t <= gto1.exponent().i + gto2.exponent().i; ++t) {
      for (int u = 0; u <= gto1.exponent().j + gto2.exponent().j; ++u) {
        for (int v = 0; v <= gto1.exponent().k + gto2.exponent().k; ++v) {
          for (int tau = 0; tau <= gto3.exponent().i + gto4.exponent().i; ++tau) {
            for (int nu = 0; nu <= gto3.exponent().j + gto4.exponent().j; ++nu) {
              for (int phi = 0; phi <= gto3.exponent().k + gto4.exponent().k; ++phi) {
                result += ((tau + nu + phi) % 2 == 0 ? 1 : -1) *
                  gto1.A().overlap(gto2.A(), t) *
                  gto1.B().overlap(gto2.B(), u) *
                  gto1.C().overlap(gto2.C(), v) *
                  gto3.A().overlap(gto4.A(), tau) *
                  gto3.B().overlap(gto4.B(), nu) *
                  gto3.C().overlap(gto4.C(), phi) *
                  HermiteAuxiliary::hermite_R(t + tau, u + nu, v + phi, 0, alpha, P - Q);
              }
            }
          }
        }
      }
    }

    double prefactor = 2.0 * std::pow(M_PI, 2.5) / (p * q * std::sqrt(p + q));
    double total_norm = gto1.norm() * gto2.norm() * gto3.norm() * gto4.norm();
    double total_coefficient = gto1.coefficient() * gto2.coefficient() * gto3.coefficient() * gto4.coefficient();
    return prefactor * total_norm * total_coefficient * result;
  }

  constexpr double alpha() const noexcept {
    return m_alpha;
  }

  constexpr Exponent exponent() const noexcept {
    return m_exponent;
  }

  constexpr Vec3 center() const noexcept {
    return m_center;
  }

  constexpr double norm() const noexcept {
    return m_norm;
  }

  constexpr double coefficient() const noexcept {
    return m_coefficient;
  }

  constexpr Gaussian A() const noexcept {
    return m_A;
  }

  constexpr Gaussian B() const noexcept {
    return m_B;
  }

  constexpr Gaussian C() const noexcept {
    return m_C;
  }

  friend std::ostream& operator<<(std::ostream& os, const GaussianTypeOrbital& g);

private:
  constexpr double unnormalized_overlap(const GaussianTypeOrbital& other) const noexcept {
    double prefactor = std::pow(M_PI / (m_alpha + other.m_alpha), 1.5);
    double total_overlap = m_A.overlap(other.m_A) * m_B.overlap(other.m_B) * m_C.overlap(other.m_C);
    return prefactor * total_overlap;
  }

  constexpr GaussianTypeOrbital with_incremented_exponents(int di, int dj, int dk) const noexcept {
    Exponent new_exponent{ m_exponent.i + di, m_exponent.j + dj, m_exponent.k + dk };
    return GaussianTypeOrbital(m_coefficient, m_center, new_exponent, m_alpha);
  }

  static constexpr double compute_norm(double alpha, int i, int j, int k) noexcept {
    double prefactor = std::pow(2.0 * alpha / M_PI, 0.75);
    double numerator = std::pow(4.0 * alpha, 0.5 * (i + j + k));
    double denominator = std::sqrt(double_factorial(2 * i - 1) * double_factorial(2 * j - 1) * double_factorial(2 * k - 1));
    return prefactor * numerator / denominator;
  }

  double m_coefficient;
  Vec3 m_center;
  Exponent m_exponent;
  double m_alpha;
  double m_norm;
  Gaussian m_A, m_B, m_C;
};

class ContractedGaussianTypeOrbital {
public:
  struct Param {
    double coefficient;
    double alpha;
  };

  constexpr ContractedGaussianTypeOrbital(const Vec3& origin, const GaussianTypeOrbital::Exponent& exponent, const std::vector<Param>& params) {
    for (const auto& p : params) {
      m_gtos.push_back({p.coefficient, origin, exponent, p.alpha});
    }
  }

  constexpr double eval(Vec3 r) const noexcept {
    double result = 0;
    for (const auto& gto : m_gtos) {
      result += gto.eval(r);
    }
    return result;
  }

  constexpr double overlap(const ContractedGaussianTypeOrbital& other) const noexcept {
    double result = 0;
    for (const auto& gto1 : m_gtos) {
      for (const auto& gto2 : other.m_gtos) {
        result += gto1.overlap(gto2);
      }
    }
    return result;
  }

  constexpr double kinetic(const ContractedGaussianTypeOrbital& other) const noexcept {
    double result = 0;
    for (const auto& gto1 : m_gtos) {
      for (const auto& gto2 : other.m_gtos) {
        result += gto1.kinetic(gto2);
      }
    }
    return result;
  }

  constexpr double nuclear_attraction(const ContractedGaussianTypeOrbital& other, const Vec3& nucleus_center) const noexcept {
    double result = 0;
    for (const auto& gto1 : m_gtos) {
      for (const auto& gto2 : other.m_gtos) {
        result += gto1.nuclear_attraction(gto2, nucleus_center);
      }
    }
    return result;
  }

  static constexpr double electron_repulsion(const ContractedGaussianTypeOrbital& cgto1, const ContractedGaussianTypeOrbital& cgto2, const ContractedGaussianTypeOrbital& cgto3, const ContractedGaussianTypeOrbital& cgto4) {
    double result = 0;
    for (const auto& gto1 : cgto1.gtos()) {
      for (const auto& gto2 : cgto2.gtos()) {
        for (const auto& gto3 : cgto3.gtos()) {
          for (const auto& gto4 : cgto4.gtos()) {
            result += GaussianTypeOrbital::electron_repulsion(gto1, gto2, gto3, gto4);
          }
        }
      }
    }
    return result;
  }

  const std::vector<GaussianTypeOrbital>& gtos() const {
    return m_gtos;
  }

  friend std::ostream& operator<<(std::ostream& os, const ContractedGaussianTypeOrbital& g);

private:
  std::vector<GaussianTypeOrbital> m_gtos;
};

