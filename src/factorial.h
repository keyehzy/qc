#pragma once

#include <cstdint>

static constexpr uint64_t factorial(uint32_t n) noexcept {
    uint64_t result = 1;
    for (uint32_t i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
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
