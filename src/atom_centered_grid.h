#pragma once

namespace SCF_LDA {
namespace atom_centered_grid {
struct GridPoint {
    Vec3 r;
    double w;
    int atom_index;
};

struct XC_Grid {
    std::vector<GridPoint> points;
    Eigen::MatrixXd phi;
};

XC_Grid build_xc_grid(const std::vector<Atom>& atoms, const std::vector<ContractedGaussianTypeOrbital>& orbitals);    
} // namespace atom_centered_grid
} // namespace SCF_LDA