#pragma once

#include "input_integrals.h"
#include "uniform_grid.h"

#include <eigen3/Eigen/Dense>

namespace SCF_LDA {
using namespace uniform_grid;

struct Result {
  double energy;
  double E_xc;
  Eigen::MatrixXd C;
  Eigen::VectorXd epsilon;
  Eigen::MatrixXd P;
};

Result run_scf(const InputIntegrals& input, const XC_Grid& xc, int n_electrons);
} // namespace SCF_LDA