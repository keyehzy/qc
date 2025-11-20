#pragma once

#include <string>
#include <vector>
#include <map>

struct ElectronShell {
  std::vector<int> angular_momentum;
  std::vector<double> alphas;
  std::vector<std::vector<double>> coefficients;
};

struct Atom {
  std::string name;
  int number;
  Vec3 center;
};

using BasisSet = std::map<std::string, std::vector<ElectronShell>>;
