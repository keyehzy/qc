#pragma once

#include <string>
#include <vector>
#include <map>

struct Shell {
  std::vector<int> angular_momentum;
  std::vector<double> alphas;
  std::vector<std::vector<double>> coefficients;
};

struct Atom {
  int number;
  Vec3 center;
};

using BasisSet = std::map<int, std::vector<Shell>>;
