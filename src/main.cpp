#include <iostream>
#include <cassert>
#include <algorithm>

#include "factorial.h"
#include "hermite_aux.h"
#include "orbital.h"
#include "vec3.h"

#include "basis_set.h"
#include "basis_set/6-31g.h"
#include "hartree_fock.h"

#include <fstream>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

std::vector<Atom> parse_molecule(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

  std::string line;
  std::getline(file, line); // Skip first line

  std::vector<Atom> atoms;

  while (std::getline(file, line)) {
    std::istringstream iss(line);
    double z;
    double v1, v2, v3;

    if (iss >> z >> v1 >> v2 >> v3) {
        atoms.push_back({(int)z, Vec3{v1,v2,v3}});
    }
  }

  return atoms;
}


int main() {
  auto molecule = parse_molecule("./assets/benzene_geom.txt");
  auto orbitals = convert(molecule, BS_6_31G);

  int n_electrons = 6 * 6 + 6;
  auto integrals = HartreeFock::InputIntegrals(molecule, orbitals);
  auto result = HartreeFock::run_scf(integrals, n_electrons);
  return 0;
}

