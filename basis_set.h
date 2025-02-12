#pragma once

#include <string>
#include <vector>
#include <map>

struct ElectronShell {
  std::vector<int> angular_momentum;
  std::vector<float> alphas;
  std::vector<std::vector<float>> coefficients;
};

struct Atom {
  std::string name;
  Vec3 center;
};

using BasisSet = std::map<std::string, std::vector<ElectronShell>>;
