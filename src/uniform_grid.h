#pragma once

#include "orbital.h"
#include "basis_set.h"

#include <eigen3/Eigen/Dense>

namespace SCF_LDA {
namespace uniform_grid {

struct GridPoint {
    Vec3 r;
    double w;
};

struct XC_Grid {
    std::vector<GridPoint> points;
    Eigen::MatrixXd phi;
};

XC_Grid build_xc_grid(const std::vector<Atom>& atoms, const std::vector<ContractedGaussianTypeOrbital>& orbitals);
} // namespace uniform_grid
} // namespace SCF_LDA