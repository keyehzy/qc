#include "basis_set.h"

std::ostream& operator<<(std::ostream& os, const Atom& a) {
  os << "Atom(" << a.number << ", Vec3(" << a.center.x << ", " << a.center.y << ", " << a.center.z << "))";
  return os;
}

std::vector<GaussianTypeOrbital::Exponent> find_exponents(int N) {
  std::vector<GaussianTypeOrbital::Exponent> result;
  result.reserve((N + 1) * (N + 2) / 2);
  for (int i = 0; i <= N; ++i) {
    for (int j = 0; j <= N - i; ++j) {
      int k = N - i - j;
      result.push_back({i, j, k});
    }
  }
  std::reverse(result.begin(), result.end());
  return result;
}

std::vector<ContractedGaussianTypeOrbital> convert(const std::vector<Atom>& molecule, const BasisSet& basis_set) {
  std::vector<ContractedGaussianTypeOrbital> result;
  for(const auto& atom : molecule) {
    for (const auto& shell : basis_set.at(atom.number)) {
      for (int a : shell.angular_momentum) {
        for (const auto& exponent : find_exponents(a)) {
          std::vector<ContractedGaussianTypeOrbital::Param> params;
          for (size_t j = 0; j < shell.alphas.size(); j++) {
            params.push_back({shell.coefficients[a][j], shell.alphas[j]});
          }
          result.push_back({atom.center, exponent, params});
        }
      }
    }
  }
  return result;
}