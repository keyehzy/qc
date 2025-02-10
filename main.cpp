#include <iostream>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <ostream>

static constexpr uint64_t factorial(uint32_t n) noexcept {
  return (n <= 1) ? 1 : n * factorial(n - 1);
}

static constexpr uint64_t double_factorial(int32_t n) noexcept {
  if(n <= 1) {
    return 1;
  }

  uint64_t result = 1;
  for (int32_t i = n; i > 1; i -= 2) {
    result *= i;
  }
  return result;
}

class Gaussian {
public:
  constexpr Gaussian(float center, int exponent, float alpha) :
    m_center(center), m_exponent(exponent), m_alpha(alpha) {}

  constexpr float eval(float x) const noexcept {
    float r = x - m_center;
    return std::pow(r, m_exponent) * std::exp(-m_alpha * r * r);
  }

  constexpr float overlap(const Gaussian& other) const noexcept {
    return hermite_iterative(m_exponent, other.m_exponent, 0, m_center - other.m_center, m_alpha, other.m_alpha);
  }

  friend std::ostream& operator<<(std::ostream& os, const Gaussian& g) {
    os << "Gaussian(center: " << g.m_center << ", exponent: " << g.m_exponent << ", alpha: " << g.m_alpha << ")";
    return os;
  }

private:

  static constexpr float hermite(int exponent1, int exponent2, int nodes, float Q, float alpha1, float alpha2) noexcept {
    float p = alpha1 + alpha2;
    float q = (alpha1 * alpha2) / p;
    if (nodes < 0 || nodes > exponent1 + exponent2)
      return 0.0f;
    else if (exponent1 == 0 && exponent2 == 0 && nodes == 0) {
      return std::exp(-q * Q * Q);
    } else if (exponent2 == 0) {
      return (1.0f / (2.0f * p)) * hermite(exponent1 - 1, exponent2, nodes - 1, Q, alpha1, alpha2) - (q * Q / alpha1) * hermite(exponent1 - 1, exponent2, nodes, Q, alpha1, alpha2) + (nodes + 1) * hermite(exponent1 - 1, exponent2, nodes + 1, Q, alpha1, alpha2);
    } else {
      return (1.0f / (2.0f * p)) * hermite(exponent1, exponent2 - 1, nodes - 1, Q, alpha1, alpha2) - (q * Q / alpha1) * hermite(exponent1, exponent2 - 1, nodes, Q, alpha1, alpha2) + (nodes + 1) * hermite(exponent1, exponent2 - 1, nodes + 1, Q, alpha1, alpha2);
    }
  }

  static constexpr float hermite_iterative(int exponent1, int exponent2, int nodes,
                                 float Q, float alpha1, float alpha2) noexcept {
    if (nodes < 0 || nodes > exponent1 + exponent2) {
      return 0.0f;
    }

    float p = alpha1 + alpha2;
    float q = (alpha1 * alpha2) / p;

    float H[exponent1 + 1][exponent2 + 1][(exponent1 + exponent2) + 1] = {0};
    std::memset(H, 0, sizeof H);

    H[0][0][0] = std::exp(-q * Q * Q);

    for (int i = 0; i <= exponent1; i++) {
      for (int j = 0; j <= exponent2; j++) {
        for (int n = 0; n <= i + j; n++) {

          if (i == 0 && j == 0 && n == 0) {
            continue;
          }

          if (j == 0 && i > 0) {
            float left  = (n - 1 >= 0) ? H[i-1][0][n-1] : 0.0f;
            float mid   = H[i-1][0][n];
            float right = (n + 1 <= (i - 1)) ? H[i-1][0][n+1] : 0.0f;
            H[i][0][n] = (1.0f / (2.0f * p)) * left - (q * Q / alpha1)  * mid + (n + 1) * right;
          } else if (j > 0) {
            float left  = (n - 1 >= 0) ? H[i][j-1][n-1] : 0.0f;
            float mid   = H[i][j-1][n];
            float right = (n + 1 <= (i + j - 1)) ? H[i][j-1][n+1] : 0.0f;
            H[i][j][n] = (1.0f / (2.0f * p)) * left - (q * Q / alpha1)  * mid + (n + 1) * right;
          }
        }
      }
    }

    return H[exponent1][exponent2][nodes];
  }

  float m_center;
  int m_exponent;
  float m_alpha;
};

struct Vec3 {
  float x, y, z;
};

struct Exponent {
  int i, j, k;
};

class GTO {
public:
  constexpr GTO(float coefficient, Vec3 center, Exponent exponent, float alpha) :
    m_coefficient(coefficient), m_center(center), m_exponent(exponent), m_alpha(alpha),
    m_norm(compute_norm(alpha, exponent.i, exponent.j, exponent.k)),
    m_A(center.x, exponent.i, alpha), m_B(center.y, exponent.j, alpha), m_C(center.z, exponent.k, alpha) {}

  constexpr float eval(Vec3 r) const noexcept {
    return m_norm * m_coefficient * m_A.eval(r.x) * m_B.eval(r.y) * m_C.eval(r.z);
  }

  constexpr float overlap(const GTO& other) const noexcept {
    return m_norm * m_norm * m_coefficient * other.m_coefficient * std::pow(M_PI / (m_alpha + other.m_alpha), 1.5f) * m_A.overlap(other.m_A) * m_B.overlap(other.m_B) * m_C.overlap(other.m_C);
  }

  constexpr float kinect(const GTO& other) const noexcept {
    float coefficient = other.m_coefficient;
    Vec3 center = other.m_center;
    auto [i,j,k] = other.m_exponent;
    float alpha = other.m_alpha;

    float term1 = alpha * (2.0f * (i + j + k) + 3.0f) *
      overlap(GTO(coefficient, center, Exponent{i,j,k}, alpha));

    float term2 = -2.0f * std::pow(alpha, 2.0f) *
      (overlap(GTO(coefficient, center, Exponent{i+2,j,k}, alpha)) +
       overlap(GTO(coefficient, center, Exponent{i,j+2,k}, alpha)) +
       overlap(GTO(coefficient, center, Exponent{i,j,k+2}, alpha)));

    float term3 = -0.5f * (i * (i - 1) * overlap(GTO(coefficient, center, Exponent{i-2,j,k}, alpha)) +
                           j * (j - 1) * overlap(GTO(coefficient, center, Exponent{i,j-2,k}, alpha)) +
                           k * (k - 1) * overlap(GTO(coefficient, center, Exponent{i,j,k-2}, alpha)));

    return term1 + term2 + term3;
  }

  friend std::ostream& operator<<(std::ostream& os, const GTO& g) {
    os << "GTO(coefficient: " << g.m_coefficient << ", alpha: " << g.m_alpha << ", [A: " << g.m_A << ", B: " << g.m_B << ", C: " << g.m_C << "])";
    return os;
  }

private:

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

struct ContractedParams {
  float coefficient;
  float alpha;
};

class ContractedGTO {
public:
  ContractedGTO(const Vec3& origin, const Exponent& exponent, const std::vector<ContractedParams>& params) {
    for (const auto& p : params) {
       m_gtos.emplace_back(p.coefficient, origin, exponent, p.alpha);
    }
  }

  float eval(Vec3 r) const noexcept {
    float result = 0;
    for (const auto& gto : m_gtos) {
      result += gto.eval(r);
    }
    return result;
  }

  float overlap(const ContractedGTO& other) const noexcept {
    float result = 0;
    for (const auto& gto1 : m_gtos) {
      for (const auto& gto2 : other.m_gtos) {
        result += gto1.overlap(gto2);
      }
    }
    return result;
  }

  float kinect(const ContractedGTO& other) const noexcept {
    float result = 0;
    for (const auto& gto1 : m_gtos) {
      for (const auto& gto2 : other.m_gtos) {
        result += gto1.kinect(gto2);
      }
    }
    return result;
  }

  friend std::ostream& operator<<(std::ostream& os, const ContractedGTO& g) {
    for (const auto& gto : g.m_gtos) {
      os << gto << "\n";
    }
    return os;
  }

private:
  std::vector<GTO> m_gtos;
};

int main() {
  auto g1 = GTO(1.0f, Vec3{0,0,0}, Exponent{0,0,0}, 1.0f);
  std::cout << g1.overlap(g1) << std::endl;
  return 0;
}
