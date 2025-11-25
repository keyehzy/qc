#pragma once

#include "../input_integrals.h"

#include <eigen3/Eigen/Dense>

namespace SCF_HF {
namespace unrestricted {
struct Result {
  double energy;

  struct {
    Eigen::MatrixXd C;
    Eigen::VectorXd epsilon;
    Eigen::MatrixXd P;
  } alpha, beta;
};

// multiplicity = 1 (Singlet), 2 (Doublet), 3 (Triplet)
Result run_scf(const InputIntegrals& input, int n_electrons, int multiplicity);
} // namespace unrestricted
} // namespace SCF_HF