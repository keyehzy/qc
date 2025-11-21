#include <iostream>
#include <cassert>
#include <algorithm>

#include "factorial.h"
#include "hermite_aux.h"
#include "orbital.h"
#include "vec3.h"

#include "basis_set.h"
#include "basis_set/sto_3g_simplified.h"

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



int main() {
  std::vector<Atom> H2O {
    Atom{8, Vec3{0, -0.143225816552, 0}},
    Atom{1, Vec3{1.638036840407, 1.136548822547, -0.96104039}},
    Atom{1, Vec3{-1.638036840407, 1.136548822547, -0.96104039}},
  };

  auto orbitals = cgto_from_basis_set(H2O, STO_3G);

  std::cout << "Overlap:\n";
  for (const auto& orbital1 : orbitals) {
    for (const auto& orbital2 : orbitals) {
        std::printf("%.04f ", orbital1.overlap(orbital2));
    }
    std::cout << "\n";
  }

  int count = 0;
  std::cout << "Electron repulsion:\n";
  for (const auto& orbital1 : orbitals) {
    for (const auto& orbital2 : orbitals) {
      for (const auto& orbital3 : orbitals) {
        for (const auto& orbital4 : orbitals) {
          std::printf("%d %.15f\n", count++, ContractedGaussianTypeOrbital::electron_repulsion(orbital1, orbital2, orbital3, orbital4));
        }
      }
    }
  }


    std::cout << "Kinetic:\n";
 for (size_t i = 0; i < orbitals.size(); i++) {
    for (size_t j = 0; j < orbitals.size(); j++) {
        std::printf("%.04f ", orbitals[i].kinetic(orbitals[j]));
    }
    std::cout << "\n";
  }

  std::cout << "Nuclear attraction:\n";
 for (size_t i = 0; i < orbitals.size(); i++) {
    for (size_t j = 0; j < orbitals.size(); j++) {
      double result = 0;
      for (const Atom& atom : H2O) {
        result += -atom.number * orbitals[i].nuclear_attraction(orbitals[j], atom.center);
      }
        std::printf("%.04f ", result);
    }
    std::cout << "\n";
  }

  return 0;
}

