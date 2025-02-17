#include <iostream>
#include <cassert>
#include <algorithm>

#include "factorial.h"
#include "hermite_aux.h"
#include "orbital.h"
#include "vec3.h"

#include "basis_set.h"
#include "basis_set/sto_3g.h"

// https://chemistry.montana.edu/callis/courses/chmy564/460water.pdf

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

std::vector<ContractedGaussianTypeOrbital> cgto_from_basis_set(const std::vector<Atom>& molecule, const BasisSet& basis_set) {
  std::vector<ContractedGaussianTypeOrbital> result;
  for(const auto& atom : molecule) {
    for (const auto& shell : basis_set.at(atom.name)) {
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

int main() {
  std::vector<Atom> H2O {
    Atom{"H", 1, Vec3{0, 1.43233673, -0.96104039}},
    Atom{"H", 1, Vec3{0, -1.43233673, -0.96104039}},
    Atom{"O", 8, Vec3{0, 0, 0.24026010}},
  };

  auto orbitals = cgto_from_basis_set(H2O, STO_3G);

  for (auto orbital : orbitals) {
    std::cout << orbital << std::endl;
  }

  std::cout << std::fixed;
  for (size_t i = 0; i < orbitals.size(); i++) {
    for (size_t j = 0; j < orbitals.size(); j++) {
      float result = 0;
      for (const Atom& atom : H2O) {
        result += -atom.number * orbitals[i].nuclear(orbitals[j], atom.center);
      }
      std::cout << result << " ";
    }
    std::cout << "\n";
  }
}

