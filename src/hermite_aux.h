#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <gsl/gsl_sf_hyperg.h>

#include "vec3.h"
#include "factorial.h"

struct HermiteAuxiliary {
  // Recursive version for reference
  static constexpr float hermite_E_(int exponent1, int exponent2, int nodes,
                                           float expTerm, float inv2p,
                                           float qQalpha1, float qQalpha2) noexcept {
    if (nodes < 0 || nodes > exponent1 + exponent2) {
      return 0.0f;
    } else if (exponent1 == 0 && exponent2 == 0 && nodes == 0) {
      return expTerm;
    } else if (exponent2 == 0) {
      float left = hermite_E_(exponent1 - 1, exponent2, nodes - 1, expTerm, inv2p, qQalpha1, qQalpha2);
      float mid = hermite_E_(exponent1 - 1, exponent2, nodes, expTerm, inv2p, qQalpha1, qQalpha2);
      float right = hermite_E_(exponent1 - 1, exponent2, nodes + 1, expTerm, inv2p, qQalpha1, qQalpha2);
      return inv2p * left - qQalpha1 * mid + (nodes + 1) * right;
    } else {
      float left = hermite_E_(exponent1, exponent2 - 1, nodes - 1, expTerm, inv2p, qQalpha1, qQalpha2);
      float mid = hermite_E_(exponent1, exponent2 - 1, nodes, expTerm, inv2p, qQalpha1, qQalpha2);
      float right = hermite_E_(exponent1, exponent2 - 1, nodes + 1, expTerm, inv2p, qQalpha1, qQalpha2);
      return inv2p * left + qQalpha2 * mid + (nodes + 1) * right;
    }
  }

static constexpr float hermite_E(int exponent1, int exponent2, int nodes,
                                          float Q, float alpha1, float alpha2) noexcept {
  float p = alpha1 + alpha2;
  float q = (alpha1 * alpha2) / p;
  float inv2p = 1.0f / (2.0f * p);
  float qQalpha1 = q * Q / alpha1;
  float qQalpha2 = q * Q / alpha2;
  float expTerm = std::exp(-q * Q * Q);

  return hermite_E_(exponent1, exponent2, nodes, expTerm, inv2p, qQalpha1, qQalpha2);
}

  static double hyp1f1(double a, double b, double x) noexcept {
    // Naive version is very bad for large values
    // Use this in the future https://arxiv.org/abs/1407.7786
    return  gsl_sf_hyperg_1F1(a, b, x);
  }

  static float boys(int n, float T) {
#if 0
    if (T > 10.0f) {
      float a = double_factorial(2 * n - 1);
      float b = std::pow(2, n + 1);
      float c = std::pow(T,  n + 0.5f);
      return a * std::sqrt(M_PI) / b / c;
    }
#endif

#if 0
    if (T < 0.5f) {
      float a = n + 0.5f;
      float b = n + 1.5f;
      float num = 2.0f * b * (1.0f + b) + b * T - a * (2.0f + b) * T;
      float denom = 2.0f * b * (1.0f + b) + (1.0f + a) * b * T;
      return num / denom / (2.0f * n + 1.0f);
    }
#endif

    return hyp1f1(n + 0.5f, n + 1.5f, -T) / (2.0f * n + 1.0f);
  }

  // Recursive version for reference
  static float hermite_R_(int i, int j, int k, int order, float p, const Vec3 &P, float T) noexcept {
    float result = 0.0f;

    if (i == 0 && j == 0 && k == 0) {
      result += std::pow(-2.0f * p, order) * boys(order, T);
    } else if (i == 0 && j == 0) {
      if (k > 1) {
        result += (k - 1) * hermite_R_(i, j, k - 2, order + 1, p, P, T);
      }
      result += P.z * hermite_R_(i, j, k - 1, order + 1, p, P, T);
    } else if (i == 0) {
      if (j > 1) {
        result += (j - 1) * hermite_R_(i, j - 2, k, order + 1, p, P, T);
      }
      result += P.y * hermite_R_(i, j - 1, k, order + 1, p, P, T);
    } else {
      if (i > 1) {
        result += (i - 1) * hermite_R_(i - 2, j, k, order + 1, p, P, T);
      }
      result += P.x * hermite_R_(i - 1, j, k, order + 1, p, P, T);
    }

    return result;
  }

  static float hermite_R(int i, int j, int k, int order, float p, const Vec3 &P) noexcept {
    float T = p * P.norm2();
    return hermite_R_(i, j, k, order, p, P, T);
  }
};
