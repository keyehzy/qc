#include "lda.h"

#include <iostream>
#include <iomanip>

namespace SCF_LDA {
std::vector<GridPoint> build_uniform_grid(const std::vector<Atom>& atoms, double padding = 3.0, int nperdim = 10) {
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

Eigen::VectorXd compute_density_on_grid(const Eigen::MatrixXd& P,
                                        const XC_Grid& xc) {
    int n_grid = static_cast<int>(xc.points.size());
    int norb   = static_cast<int>(P.rows());
    Eigen::VectorXd rho(n_grid);
    for (int g = 0; g < n_grid; ++g) {
        double r = 0.0;
        for (int mu = 0; mu < norb; ++mu) {
            double phi_mu = xc.phi(g, mu);
            for (int nu = 0; nu < norb; ++nu) {
                r += P(mu, nu) * phi_mu * xc.phi(g, nu);
            }
        }
        rho(g) = r;
    }
    return rho;
}

struct LDA_X_Only {
    static double eps_x(double rho) {
        if (rho <= 0.0) return 0.0;
        static const double Cx = -0.75 * std::pow(3.0 / M_PI, 1.0 / 3.0);
        return Cx * std::cbrt(rho); // ρ^{1/3}
    }

    static double v_x(double rho) {
        if (rho <= 0.0) return 0.0;
        return (4.0 / 3.0) * eps_x(rho);
    }
};

double eval_xc(const Eigen::VectorXd& rho, const XC_Grid& xc, Eigen::VectorXd& eps_xc_grid, Eigen::VectorXd& v_xc_grid) {

    int n_grid = static_cast<int>(xc.points.size());
    eps_xc_grid.resize(n_grid);
    v_xc_grid.resize(n_grid);
    double E_xc_total = 0.0;

    for (int g = 0; g < n_grid; ++g) {
        double r = std::max(rho(g), 1e-14); // avoid ρ=0
        double eps_xc = LDA_X_Only::eps_x(r);
        double v_xc   = LDA_X_Only::v_x(r);

        eps_xc_grid(g) = eps_xc;
        v_xc_grid(g)   = v_xc;

        E_xc_total += xc.points[g].w * r * eps_xc;
    }

    return E_xc_total;
}

Eigen::MatrixXd build_V_xc(const XC_Grid& xc, const Eigen::VectorXd& v_xc_grid, int norb) {

    int n_grid = static_cast<int>(xc.points.size());
    Eigen::MatrixXd V_xc = Eigen::MatrixXd::Zero(norb, norb);

    for (int mu = 0; mu < norb; ++mu) {
        for (int nu = 0; nu < norb; ++nu) {
            double sum = 0.0;
            for (int g = 0; g < n_grid; ++g) {
                double w   = xc.points[g].w;
                double vxc = v_xc_grid(g);
                double phi_mu = xc.phi(g, mu);
                double phi_nu = xc.phi(g, nu);
                sum += w * vxc * phi_mu * phi_nu;
            }
            V_xc(mu, nu) = sum;
        }
    }
    return V_xc;
}

Result run_scf(const InputIntegrals& input, const XC_Grid& xc, int n_electrons) {
    
    const int norb = input.S.rows();
    const int n_occ = n_electrons / 2;
    const double convergence = 1e-6;
    const int max_iter = 50;
    
    // Step 1: Core Hamiltonian & Orthogonalizer
    const Eigen::MatrixXd H_core = input.T + input.V;
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_S(input.S);
    
    if (eig_S.eigenvalues().minCoeff() < 1e-10) {
        throw std::runtime_error("Overlap matrix is near-singular!");
    }
    
    const Eigen::MatrixXd X = eig_S.eigenvectors() * eig_S.eigenvalues().cwiseSqrt().cwiseInverse().asDiagonal() * eig_S.eigenvectors().transpose();
    
    // Step 2: Initial guess from H_core
    Eigen::MatrixXd H_ortho = X.transpose() * H_core * X;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_H(H_ortho);
    Eigen::MatrixXd C = X * eig_H.eigenvectors();
    Eigen::MatrixXd P = 2.0 * C * C.transpose();

    // Step 3: SCF iterations
    Eigen::MatrixXd P_prev = P;
    double E_total_prev = 0.0;
    
    std::cout << "\nStarting SCF iterations...\n" << std::endl;
    
    for (int iter = 0; iter < max_iter; ++iter) {
      // Map to Tensor
      Eigen::TensorMap<Eigen::Tensor<double, 2>> P_tensor(P.data(), norb, norb);
      
      // Coulomb J
      // J_ij = Sum_kl (ij|kl) * P_kl
      // Contract ERI dims 2,3 with P dims 0,1
      Eigen::array<Eigen::IndexPair<int>, 2> j_contract = {
        Eigen::IndexPair<int>(2, 0),  // k
        Eigen::IndexPair<int>(3, 1)   // l
      };
      Eigen::Tensor<double, 2> J_tensor = input.ERI.contract(P_tensor, j_contract);
      Eigen::MatrixXd J = Eigen::Map<Eigen::MatrixXd>(J_tensor.data(), norb, norb);

      // XC
      Eigen::VectorXd rho_grid = compute_density_on_grid(P, xc);
      Eigen::VectorXd eps_xc_grid, v_xc_grid;
      double E_xc = eval_xc(rho_grid, xc, eps_xc_grid, v_xc_grid);
      Eigen::MatrixXd V_xc = build_V_xc(xc, v_xc_grid, norb);
      
      // Build Fock matrix
      Eigen::MatrixXd F = H_core + J + V_xc;
      
      // Diagonalize Fock
      const Eigen::MatrixXd F_ortho = X.transpose() * F * X;
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_F(F_ortho);
      const Eigen::MatrixXd C_ortho = eig_F.eigenvectors();
      const Eigen::VectorXd epsilon = eig_F.eigenvalues();
      C = X * C_ortho;
      
      // New density
      const auto C_occ = C.leftCols(n_occ);
      P = 2.0 * C_occ * C_occ.transpose();
      
      // Energy
      double E_one_e = (P.cwiseProduct(H_core)).sum();
      double E_coul  = 0.5 * (P.cwiseProduct(J)).sum();
      double E_total = E_one_e + E_coul + E_xc + input.nuc_repulsion;
      
      // Convergence
      const double rmsd = (P - P_prev).norm();
      const double deltaE = std::abs(E_total - E_total_prev);
      
      std::cout << "Iter " << std::setw(3) << iter + 1 
                << ": E = " << std::fixed << std::setprecision(10) << E_total
                << ", ΔE = " << std::scientific << deltaE
                << ", RMSD = " << rmsd << std::endl;
      
      if (rmsd < convergence && deltaE < convergence) {
          std::cout << "\nSCF converged in " << iter + 1 << " iterations!" << std::endl;
          return {E_total, E_xc, C, epsilon, P};
      }
      
      E_total_prev = E_total;
      P_prev = P;
  }
  
  throw std::runtime_error("SCF failed to converge!");
}
} // namespace SCF_LDA