#include <iostream>
#include <cassert>
#include <unordered_map>

#include "factorial.h"
#include "hermite_aux.h"
#include "orbital.h"
#include "vec3.h"

// https://content.wolfram.com/sites/19/2012/02/Ho.pdf

int main() {
  // STO-3G basis set
  constexpr size_t basis_size = 3;

  std::vector<Vec3> R{{0, 1.43233673, -0.96104039},
                      {0, -1.43233673, -0.96104039},
                      {0, 0, 0.24026010}};

  std::vector<std::vector<float>> coefficients{
    {0.1543289673, 0.5353281423, 0.4446345422},
    {0.1543289673, 0.5353281423, 0.4446345422},
    {0.1543289673, 0.5353281423, 0.4446345422},
    {-0.09996722919, 0.3995128261, 0.7001154689},
    {0.155916275, 0.6076837186, 0.3919573931},
    {0.155916275, 0.6076837186, 0.3919573931},
    {0.155916275, 0.6076837186, 0.3919573931}};

  std::vector<std::vector<float>> alphas{
    {3.425250914, 0.6239137298, 0.168855404},
    {3.425250914, 0.6239137298, 0.168855404},
    {130.7093214, 23.80886605, 6.443608313},
    {5.033151319, 1.169596125, 0.38038896} ,
    {5.033151319, 1.169596125, 0.38038896},
    {5.033151319, 1.169596125, 0.38038896},
    {5.033151319, 1.169596125, 0.38038896}};

  size_t data_size = coefficients.size();
  assert(data_size == alphas.size());

  std::vector<std::vector<ContractedGaussianTypeOrbital::Param>> params;
  params.reserve(data_size);

  for (size_t i = 0; i < data_size; i++) {
    std::vector<ContractedGaussianTypeOrbital::Param> p;
    p.reserve(basis_size);
    for (size_t j = 0; j < basis_size; j++) {
      p.emplace_back(coefficients[i][j], alphas[i][j]);
    }
    params.emplace_back(std::move(p));
  }

  // H1s, H2s, O1s, O2s, O2px, O2py, and O2pz

  std::vector<Vec3> center{
    R[0], R[1], R[2], R[2], R[2], R[2], R[2]
  };

  std::vector<GaussianTypeOrbital::Exponent> exponents{
    {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1}
  };

  std::vector<ContractedGaussianTypeOrbital> orbitals {
    { center[0], exponents[0], params[0] },
    { center[1], exponents[1], params[1] },
    { center[2], exponents[2], params[2] },
    { center[3], exponents[3], params[3] },
    { center[4], exponents[4], params[4] },
    { center[5], exponents[5], params[5] },
    { center[6], exponents[6], params[6] }
  };

  for (size_t i = 0; i < orbitals.size(); i++) {
    for (size_t j = 0; j < orbitals.size(); j++) {
      std::cout << std::fixed;
      std::cout << orbitals[i].overlap(orbitals[j]) << " ";
    }
    std::cout << "\n";
  }

  return 0;
}
