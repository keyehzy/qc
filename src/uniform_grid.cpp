#include "uniform_grid.h"

namespace SCF_LDA {
namespace uniform_grid {
std::vector<GridPoint> build_uniform_grid(const std::vector<Atom>& atoms, double padding = 3.0, int nperdim = 50) {
    // crude bounding box around molecule
    Vec3 min = atoms[0].center;
    Vec3 max = atoms[0].center;
    for (const auto& a : atoms) {
        min.x = std::min(min.x, a.center.x);
        min.y = std::min(min.y, a.center.y);
        min.z = std::min(min.z, a.center.z);
        max.x = std::max(max.x, a.center.x);
        max.y = std::max(max.y, a.center.y);
        max.z = std::max(max.z, a.center.z);
    }

    // expand a bit
    min = {min.x - padding, min.y - padding, min.z - padding};
    max = {max.x + padding, max.y + padding, max.z + padding};

    // uniform grid
    Vec3 step{(max.x - min.x) / (nperdim - 1), (max.y - min.y) / (nperdim - 1), (max.z - min.z) / (nperdim - 1)};

    double w = step.x * step.y * step.z; // uniform volume element

    std::vector<GridPoint> grid;
    grid.reserve(nperdim * nperdim * nperdim);
    for (int ix = 0; ix < nperdim; ++ix) {
        for (int iy = 0; iy < nperdim; ++iy) {
            for (int iz = 0; iz < nperdim; ++iz) {
                Vec3 r{min.x + ix * step.x, min.y + iy * step.y, min.z + iz * step.z};
                grid.push_back({r, w});
            }
        }
    }
    return grid;
}

XC_Grid build_xc_grid(const std::vector<Atom>& atoms, const std::vector<ContractedGaussianTypeOrbital>& orbitals) {
    XC_Grid xc;
    xc.points = build_uniform_grid(atoms);
    int n_grid = static_cast<int>(xc.points.size());
    int norb   = static_cast<int>(orbitals.size());

    xc.phi.setZero(n_grid, norb);
    for (int g = 0; g < n_grid; ++g) {
        const Vec3& r = xc.points[g].r;
        for (int mu = 0; mu < norb; ++mu) {
            xc.phi(g, mu) = orbitals[mu].eval(r);
        }
    }
    return xc;
}
} // namespace uniform_grid
} // namespace SCF_LDA