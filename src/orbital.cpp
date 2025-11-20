#include "orbital.h"

std::ostream& operator<<(std::ostream& os, const Gaussian& g) {
  os << "Gaussian(center: " << g.m_center << ", exponent: " << g.m_exponent << ", alpha: " << g.m_alpha << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const GaussianTypeOrbital& g) {
  os << "GaussianTypeOrbital(norm: " << g.norm() << ", coefficient: " << g.m_coefficient << ", alpha: " << g.m_alpha << ", [A: " << g.m_A << ", B: " << g.m_B << ", C: " << g.m_C << "])";
  return os;
}

std::ostream& operator<<(std::ostream& os, const ContractedGaussianTypeOrbital& g) {
  for (const auto& gto : g.m_gtos) {
    os << gto << "\n";
  }
  return os;
}

