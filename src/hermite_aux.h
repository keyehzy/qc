#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <gsl/gsl_sf_hyperg.h>

#include "vec3.h"
#include "factorial.h"

struct HermiteAuxiliary {
  // Recursive version for reference
  static constexpr double hermite_E_(int exponent1, int exponent2, int nodes,
                                           double expTerm, double inv2p,
                                           double qQalpha1, double qQalpha2) noexcept {
    if (nodes < 0 || nodes > exponent1 + exponent2) {
      return 0.0;
    } else if (exponent1 == 0 && exponent2 == 0 && nodes == 0) {
      return expTerm;
    } else if (exponent2 == 0) {
      double left = hermite_E_(exponent1 - 1, exponent2, nodes - 1, expTerm, inv2p, qQalpha1, qQalpha2);
      double mid = hermite_E_(exponent1 - 1, exponent2, nodes, expTerm, inv2p, qQalpha1, qQalpha2);
      double right = hermite_E_(exponent1 - 1, exponent2, nodes + 1, expTerm, inv2p, qQalpha1, qQalpha2);
      return inv2p * left - qQalpha1 * mid + (nodes + 1) * right;
    } else {
      double left = hermite_E_(exponent1, exponent2 - 1, nodes - 1, expTerm, inv2p, qQalpha1, qQalpha2);
      double mid = hermite_E_(exponent1, exponent2 - 1, nodes, expTerm, inv2p, qQalpha1, qQalpha2);
      double right = hermite_E_(exponent1, exponent2 - 1, nodes + 1, expTerm, inv2p, qQalpha1, qQalpha2);
      return inv2p * left + qQalpha2 * mid + (nodes + 1) * right;
    }
  }

static constexpr double hermite_E(int exponent1, int exponent2, int nodes,
                                          double Q, double alpha1, double alpha2) noexcept {
  double p = alpha1 + alpha2;
  double q = (alpha1 * alpha2) / p;
  double inv2p = 1.0 / (2.0 * p);
  double qQalpha1 = q * Q / alpha1;
  double qQalpha2 = q * Q / alpha2;
  double expTerm = std::exp(-q * Q * Q);

  return hermite_E_(exponent1, exponent2, nodes, expTerm, inv2p, qQalpha1, qQalpha2);
}

  static double hyp1f1(double a, double b, double x) noexcept {
    // Naive version is very bad for large values
    // Use this in the future https://arxiv.org/abs/1407.7786
    return  gsl_sf_hyperg_1F1(a, b, x);
  }

  static double boys(int n, double T) {
#if 0
    if (T > 10.0) {
      double a = double_factorial(2 * n - 1);
      double b = std::pow(2, n + 1);
      double c = std::pow(T,  n + 0.5);
      return a * std::sqrt(M_PI) / b / c;
    }
#endif

#if 0
    if (T < 0.5) {
      double a = n + 0.5;
      double b = n + 1.5;
      double num = 2.0 * b * (1.0 + b) + b * T - a * (2.0 + b) * T;
      double denom = 2.0 * b * (1.0 + b) + (1.0 + a) * b * T;
      return num / denom / (2.0 * n + 1.0);
    }
#endif

    return hyp1f1(n + 0.5, n + 1.5, -T) / (2.0 * n + 1.0);
  }

  // Recursive version for reference
  static double hermite_R_(int i, int j, int k, int order, double p, const Vec3 &P, double T) noexcept {
    double result = 0.0;

    if (i == 0 && j == 0 && k == 0) {
      result += std::pow(-2.0 * p, order) * boys(order, T);
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

  static double hermite_R(int i, int j, int k, int order, double p, const Vec3 &P) noexcept {
    double T = p * P.norm2();
    return hermite_R_(i, j, k, order, p, P, T);
  }
};
