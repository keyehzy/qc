#pragma once

#include "input_integrals.h"

#include <eigen3/Eigen/Dense>

namespace SCF_LDA {
struct Result {
  double energy;
  double E_xc;
  Eigen::MatrixXd C;
  Eigen::VectorXd epsilon;
  Eigen::MatrixXd P;
};

struct GridPoint {
    Vec3 r;   // coordinate
    double w; // weight
};

struct XC_Grid {
    std::vector<GridPoint> points; // size: n_grid
    Eigen::MatrixXd phi;           // (n_grid, norb)
};

XC_Grid build_xc_grid(const std::vector<Atom>& atoms, const std::vector<ContractedGaussianTypeOrbital>& orbitals);
Result run_scf(const InputIntegrals& input, const XC_Grid& xc, int n_electrons);
} // namespace SCF_LDA