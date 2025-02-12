#pragma once

#include <vector>
#include <cmath>

struct HermiteAuxiliary {
  static constexpr int index(int i, int j, int n, int exponent1, int exponent2) noexcept {
    return (i * (exponent2 + 1) + j) * (exponent1 + exponent2 + 1) + n;
  }

  static constexpr float E(int exponent1, int exponent2, int nodes, float Q, float alpha1, float alpha2) noexcept {
    if (exponent1 < 0 || exponent2 < 0) {
      return 0.0f;
    }
    
    if (nodes < 0 || nodes > exponent1 + exponent2) {
      return 0.0f;
    }

    float p        = alpha1 + alpha2;
    float q        = (alpha1 * alpha2) / p;
    float inv2p    = 1.0f / (2.0f * p);
    float qQalpha1 = q * Q / alpha1;
    float qQalpha2 = q * Q / alpha2;

    std::vector<float> H((exponent1 + 1) * (exponent2 + 1) * (exponent1 + exponent2 + 1), 0.0f);

    H[index(0, 0, 0, exponent1, exponent2)] = std::exp(-q * Q * Q);

    for (int i = 1; i <= exponent1; i++) {
      for (int n = 0; n <= i; n++) {
        float left  = (n - 1 >= 0) ? H[index(i - 1, 0, n - 1, exponent1, exponent2)] : 0.0f;
        float mid   = H[index(i -1, 0, n, exponent1, exponent2)];
        float right = (n + 1 <= i -1) ? H[index(i -1, 0, n + 1, exponent1, exponent2)] : 0.0f;
        H[index(i, 0, n, exponent1, exponent2)] = inv2p * left - qQalpha1 * mid + (n + 1) * right;
      }
    }

    for (int j = 1; j <= exponent2; j++) {
      for (int i = 0; i <= exponent1; i++) {
        for (int n = 0; n <= i + j; n++) {
          float left  = (n - 1 >= 0) ? H[index(i, j - 1, n - 1, exponent1, exponent2)] : 0.0f;
          float mid   = H[index(i, j - 1, n, exponent1, exponent2)];
          float right = (n + 1 <= (i + j - 1)) ? H[index(i, j - 1, n + 1, exponent1, exponent2)] : 0.0f;
          H[index(i, j, n, exponent1, exponent2)] = inv2p * left + qQalpha2 * mid + (n + 1) * right;
        }
      }
    }
    return H[index(exponent1, exponent2, nodes, exponent1, exponent2)];
  }

  // Recursive version for reference
  static constexpr float hermite_E(int exponent1, int exponent2, int nodes, float Q,
                                   float alpha1, float alpha2) noexcept {
    float p        = alpha1 + alpha2;
    float q        = (alpha1 * alpha2) / p;
    float inv2p    = 1.0f / (2.0f * p);
    float qQalpha1 = q * Q / alpha1;
    float qQalpha2 = q * Q / alpha2;

    if (nodes < 0 || nodes > exponent1 + exponent2) {
      return 0.0f;
    } else if (exponent1 == 0 && exponent2 == 0 && nodes == 0) {
      return std::exp(-q * Q * Q);
    } else if (exponent2 == 0) {
      float left = hermite_E(exponent1 - 1, exponent2, nodes - 1, Q, alpha1, alpha2);
      float mid = hermite_E(exponent1 - 1, exponent2, nodes, Q, alpha1, alpha2);
      float right = hermite_E(exponent1 - 1, exponent2, nodes + 1, Q, alpha1, alpha2);
      return inv2p * left - qQalpha1 * mid  + (nodes + 1) * right;
    } else {
      float left = hermite_E(exponent1, exponent2 - 1, nodes - 1, Q, alpha1, alpha2);
      float mid = hermite_E(exponent1, exponent2 - 1, nodes, Q, alpha1, alpha2);
      float right = hermite_E(exponent1, exponent2 - 1, nodes + 1, Q, alpha1, alpha2);
      return inv2p * left + qQalpha2 * mid + (nodes + 1) * right;
    }
  }
};
