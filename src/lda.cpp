#include "lda.h"
#include "functionals.h"
#include "diis.h"

#include <iostream>
#include <iomanip>

namespace SCF_LDA {
Eigen::VectorXd compute_density_on_grid(const Eigen::MatrixXd& P, const XC_Grid& xc) {
    // 1. Temp = Phi * P (N_grid x N_orb)
    Eigen::MatrixXd Temp = xc.phi * P;

    // 2. Row-wise dot product
    // rho(g) = dot(Temp.row(g), phi.row(g))
    return (Temp.cwiseProduct(xc.phi)).rowwise().sum();
}

double eval_xc(const Eigen::VectorXd& rho, const XC_Grid& xc, Eigen::VectorXd& eps_xc_grid, Eigen::VectorXd& v_xc_grid, CorrelationFunctional corr_type = XC_VWN) {
    int n_grid = static_cast<int>(xc.points.size());
    eps_xc_grid.resize(n_grid);
    v_xc_grid.resize(n_grid);
    double E_xc_total = 0.0;

    for (int g = 0; g < n_grid; ++g) {
        double r = std::max(rho(g), 1e-14);

        // 1. Exchange (Slater)
        double ex = 0.0;
        double vx = 0.0;
        LDA_Slater::eval(r, ex, vx);

        // 2. Correlation
        double ec = 0.0;
        double vc = 0.0;

        switch(corr_type) {
        case XC_VWN:
            LDA_VWN::eval(r, ec, vc);
            break;
        case XC_PW92:
            LDA_PW92::eval(r, ec, vc);
            break;
        case XC_NONE:
        default:
            break;
        }

        // 3. Combine
        double eps_total = ex + ec;
        double v_total   = vx + vc;

        eps_xc_grid(g) = eps_total;
        v_xc_grid(g)   = v_total;

        E_xc_total += xc.points[g].w * r * eps_total;
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
    const double convergence = 1e-8;
    const int max_iter = 100;
    DIIS diis;

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
    Eigen::MatrixXd C_occ = C.leftCols(n_occ);
    Eigen::MatrixXd P = 2.0 * C_occ * C_occ.transpose();

    // Step 3: SCF iterations
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

      // XC Calculations
      Eigen::VectorXd rho_grid = compute_density_on_grid(P, xc);
      Eigen::VectorXd eps_xc_grid;
      Eigen::VectorXd v_xc_grid;
      double E_xc = eval_xc(rho_grid, xc, eps_xc_grid, v_xc_grid, XC_VWN);
      Eigen::MatrixXd V_xc = build_V_xc(xc, v_xc_grid, norb);

      // Build Fock matrix
      Eigen::MatrixXd F = H_core + J + V_xc;

      // DIIS Extrapolation
      Eigen::MatrixXd F_diis = diis.compute(F, P, input.S);

      // Energy
      double E_one_e = (P.cwiseProduct(H_core)).sum();
      double E_coul  = 0.5 * (P.cwiseProduct(J)).sum();
      double E_total = E_one_e + E_coul + E_xc + input.nuc_repulsion;

      // Diagonalize Fock
      const Eigen::MatrixXd F_ortho = X.transpose() * F_diis * X;
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_F(F_ortho);
      const Eigen::MatrixXd C_ortho = eig_F.eigenvectors();
      const Eigen::VectorXd epsilon = eig_F.eigenvalues();

      // New density
      Eigen::MatrixXd C_new = X * C_ortho;
      const auto C_occ_new = C_new.leftCols(n_occ);
      Eigen::MatrixXd P_new = 2.0 * C_occ_new * C_occ_new.transpose();

      // Convergence Check
      const double rmsd = (P_new - P).norm();
      const double deltaE = std::abs(E_total - E_total_prev);

      std::cout << "Iter " << std::setw(3) << iter + 1
                << ": E_one_e = " << std::fixed << std::setprecision(10) << E_one_e
                << ": E_coul = " << std::fixed << std::setprecision(10) << E_coul
                << ": E = " << std::fixed << std::setprecision(10) << E_total
                << ", ΔE = " << std::scientific << deltaE
                << ", RMSD = " << rmsd << std::endl;

      if (rmsd < convergence && deltaE < convergence) {
          std::cout << "\nSCF converged in " << iter + 1 << " iterations!" << std::endl;
          return {E_total, E_xc, C_new, epsilon, P};
      }

      E_total_prev = E_total;
      P = P_new;
  }

  throw std::runtime_error("SCF failed to converge!");
}
} // namespace SCF_LDA