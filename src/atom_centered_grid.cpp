#include "atom_centered_grid.h"

#include "lebedev_15.h"

namespace SCF_LDA {
namespace atom_centered_grid {
namespace {
void compute_gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w) {
    x.resize(n);
    w.resize(n);
    const double eps = 1e-14;
    int m = (n + 1) / 2;
    for (int i = 0; i < m; ++i) {
        double z = std::cos(M_PI * (i + 0.75) / (n + 0.5));
        double pp, p1, p2, p3;
        do {
            p1 = 1.0;
            p2 = 0.0;
            for (int j = 0; j < n; ++j) {
                p3 = p2;
                p2 = p1;
                p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0);
            }
            pp = n * (z * p1 - p2) / (z * z - 1.0);
            z = z - p1 / pp;
        } while (std::abs(p1 / pp) > eps);

        x[i] = -z;
        x[n - 1 - i] = z;
        w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
        w[n - 1 - i] = w[i];
    }
}
}

struct RadialPoint { 
    double r;
    double w;
};

// Map x in [-1, 1] to r in [0, inf) using r = r_scale * (1+x)/(1-x)
// Jacobian dr/dx = 2 * r_scale / (1-x)^2
// Weight_final = w_GL * Jacobian * r^2 (volume element)
std::vector<RadialPoint> build_radial_grid(int n_rad, double r_scale) {
    std::vector<double> x, w;
    compute_gauss_legendre(n_rad, x, w);
    
    std::vector<RadialPoint> grid;
    grid.reserve(n_rad);
    
    for(int i=0; i<n_rad; ++i) {
        if (x[i] >= 1.0) continue; 
        
        double one_minus_x = 1.0 - x[i];
        double one_plus_x  = 1.0 + x[i];
        
        double r = r_scale * one_plus_x / one_minus_x;
        double dr_dx = 2.0 * r_scale / (one_minus_x * one_minus_x);        
        double weight = w[i] * dr_dx * r * r;
        
        grid.push_back({r, weight});
    }
    return grid;
}

// Calculates the fuzzy cell weight for point r belonging to atom center `atom_idx`
// P_A(r) = Product_B s(mu_AB)
double get_becke_weight(const Vec3& r, int atom_idx, const std::vector<Atom>& atoms) {
    double P_A = 1.0;
    
    // Distance to current center
    double dx = r.x - atoms[atom_idx].center.x;
    double dy = r.y - atoms[atom_idx].center.y;
    double dz = r.z - atoms[atom_idx].center.z;
    double dist_A = std::sqrt(dx*dx + dy*dy + dz*dz);

    for (size_t b = 0; b < atoms.size(); ++b) {
        if (b == (size_t)atom_idx) continue;
        
        // Distance to neighbor
        double dx_b = r.x - atoms[b].center.x;
        double dy_b = r.y - atoms[b].center.y;
        double dz_b = r.z - atoms[b].center.z;
        double dist_B = std::sqrt(dx_b*dx_b + dy_b*dy_b + dz_b*dz_b);
        
        // Distance between nuclei
        double dx_ab = atoms[atom_idx].center.x - atoms[b].center.x;
        double dy_ab = atoms[atom_idx].center.y - atoms[b].center.y;
        double dz_ab = atoms[atom_idx].center.z - atoms[b].center.z;
        double R_AB = std::sqrt(dx_ab*dx_ab + dy_ab*dy_ab + dz_ab*dz_ab);
        
        // Confocal coordinate mu
        double mu = (dist_A - dist_B) / R_AB;     
        
        // Iterate
        double f = mu;
        for(int k=0; k<3; ++k) {
            f = 1.5 * f - 0.5 * f * f * f;
        }
        
        double s = 0.5 * (1.0 - f);
        P_A *= s;
    }
    return P_A;
}

// Normalizes the weights across all atoms
void apply_becke_weights(std::vector<GridPoint>& grid_points const std::vector<Atom>& atoms) {
    for (size_t i = 0; i < grid_points.size(); ++i) {
        Vec3 r = grid_points[i].r;
        
        // Calculate sum of P_B(r) for all atoms B
        double sum_P = 0.0;
        for (size_t b = 0; b < atoms.size(); ++b) {
            sum_P += get_becke_weight(r, (int)b, atoms);
        }
        
        // Calculate P_A(r) for the atom this point was generated from
        int current_atom = grid_points[i].atom_index;
        double P_A = get_becke_weight(r, current_atom, atoms);
        
        // Modified weight
        if (sum_P > 1e-14) {
            grid_points[i].w *= (P_A / sum_P);
        } else {
            grid_points[i].w = 0.0;
        }
    }
}

XC_Grid build_xc_grid(const std::vector<Atom>& atoms, const std::vector<ContractedGaussianTypeOrbital>& orbitals) {

    XC_Grid xc;
    
    const int n_rad = 40; 
    const double r_scale = 1.5;  // Bragg radius
    
    const auto& leb_grid = lebedev_15;
    auto rad_grid = build_radial_grid(n_rad, r_scale);

    // Loop over all atoms and build local grids
    for (size_t a = 0; a < atoms.size(); ++a) {
        const auto& atom = atoms[a];
        
        for (const auto& rad : rad_grid) {
            for (const auto& ang : leb_grid) {
                
                // Spherical to Cartesian relative to atom center
                double x = atom.center.x + rad.r * std::cos(ang.theta) * std::sin(ang.phi);
                double y = atom.center.y + rad.r * std::sin(ang.theta) * std::sin(ang.phi);
                double z = atom.center.z + rad.r * std::cos(ang.phi);
                
                // Combined weight = Radial_W * Angular_W
                double w = rad.w * ang.w * 4.0 * M_PI;
                
                // Pruning: Skip points with negligible weight (e.g. very far out)
                if (w < 1e-14) continue;

                xc.points.push_back({{x, y, z}, w, a});
            }
        }
    }

    // Apply Becke Partitioning to fix double-counting of overlaps
    apply_becke_weights(xc.points, atoms);

    // Evaluate Basis Functions on the new Grid
    int n_grid = static_cast<int>(xc.points.size());
    int norb   = static_cast<int>(orbitals.size());

    xc.phi.setZero(n_grid, norb);
    for (int g = 0; g < n_grid; ++g) {
        const Vec3& r = xc.points[g].r;
        // Optimization: Check distance to basis function center.
        // If too small, skip it.
        for (int mu = 0; mu < norb; ++mu) {
            xc.phi(g, mu) = orbitals[mu].eval(r);
        }
    }    
    
    return xc;
}
} // namespace atom_centered_grid
} // namespace SCF_LDA