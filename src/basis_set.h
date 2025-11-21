#pragma once

#include <string>
#include <vector>
#include <map>
#include <ostream>

#include "vec3.h"
#include "orbital.h"

struct Shell {
  std::vector<int> angular_momentum;
  std::vector<double> alphas;
  std::vector<std::vector<double>> coefficients;
};

struct Atom {
  int number;
  Vec3 center;

  friend std::ostream& operator<<(std::ostream& os, const Atom& a);
};

using BasisSet = std::map<int, std::vector<Shell>>;

std::vector<ContractedGaussianTypeOrbital> convert(const std::vector<Atom>& molecule, const BasisSet& basis_set);