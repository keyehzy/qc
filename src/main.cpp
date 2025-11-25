#include <iostream>
#include <cassert>
#include <algorithm>

#include "factorial.h"
#include "hermite_aux.h"
#include "orbital.h"
#include "vec3.h"

#include "basis_set.h"
#include "basis_set/sto-3g.h"
#include "lda.h"
#include "hartree_fock/unrestricted_hf.h"

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
  auto molecule = parse_molecule("./assets/h2o/STO-3G/geom.dat");
  auto orbitals = convert(molecule, BS_STO_3G);

  int n_electrons = 10;
  int multiplicity = 1;
  auto integrals = InputIntegrals(molecule, orbitals);
  // auto xc_grid = SCF_LDA::atom_centered_grid::build_xc_grid(molecule, orbitals);
  // auto result = SCF_LDA::run_scf(integrals, xc_grid, n_electrons);
  auto result = SCF_HF::unrestricted::run_scf(integrals, n_electrons, multiplicity);
  return 0;
}

