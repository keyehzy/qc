#pragma once

#include "../input_integrals.h"

#include <eigen3/Eigen/Dense>

namespace SCF_HF {
namespace restricted {
struct Result {
  double energy;
  Eigen::MatrixXd C;
  Eigen::VectorXd epsilon;
  Eigen::MatrixXd P;
};

Result run_scf(const InputIntegrals& input, int n_electrons);
} // namespace restricted
} // namespace SCF_HF