#include "lda.h"
#include "functionals.h"
#include "diis.h"

#include <iostream>
#include <iomanip>

namespace SCF_LDA {
namespace {
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

Eigen::MatrixXd build_density(const Eigen::MatrixXd& C, int n_occ) {
    return 2.0 * C.leftCols(n_occ) * C.leftCols(n_occ).transpose();
}

Eigen::MatrixXd compute_coulomb(const Eigen::Tensor<double, 4>& ERI, const Eigen::MatrixXd& P) {
  Eigen::TensorMap<Eigen::Tensor<double, 2>> P_map(const_cast<double*>(P.data()), P.rows(), P.cols());
  Eigen::array<Eigen::IndexPair<int>, 2> j_contr = {
      Eigen::IndexPair<int>(2, 0),
      Eigen::IndexPair<int>(3, 1)
  };
  Eigen::Tensor<double, 2> J_tensor = ERI.contract(P_map, j_contr);
  Eigen::MatrixXd J = Eigen::Map<Eigen::MatrixXd>(J_tensor.data(), P.rows(), P.cols());
  return J;
}

std::pair<double, Eigen::MatrixXd> compute_exchange_correlation(const Eigen::MatrixXd& P, const XC_Grid& xc, CorrelationFunctional corr_type = XC_VWN) {
    Eigen::VectorXd rho_grid = compute_density_on_grid(P, xc);
    Eigen::VectorXd eps_xc_grid;
    Eigen::VectorXd v_xc_grid;
    double E_xc = eval_xc(rho_grid, xc, eps_xc_grid, v_xc_grid, corr_type);
    Eigen::MatrixXd V_xc = build_V_xc(xc, v_xc_grid, P.cols());
    return {E_xc, V_xc};
}

void diagonalize_fock_and_reconstruct_answer(const Eigen::MatrixXd& F, const Eigen::MatrixXd& X, int n_occ, Eigen::MatrixXd& C, Eigen::VectorXd& eps, Eigen::MatrixXd& P) {
    const Eigen::MatrixXd F_ortho = X.transpose() * F * X;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(F_ortho);
    C = X * eig.eigenvectors();
    eps = eig.eigenvalues();
    P = build_density(C, n_occ);
}

double compute_one_e_energy(const Eigen::MatrixXd& P, const Eigen::MatrixXd& H_core) {
    return (P.cwiseProduct(H_core)).sum();
}

double compute_coul_energy(const Eigen::MatrixXd& P, const Eigen::MatrixXd& J) {
    return 0.5 * (P.cwiseProduct(J)).sum();
}
} // anonymous namespace


Result run_scf(const InputIntegrals& input, const XC_Grid& xc, int n_electrons) {
    if (n_electrons % 2 != 0) {
      throw std::runtime_error("Invalid electron count for closed shell");
    }
    
    const int n_occ = n_electrons / 2;
    const double convergence = 1e-8;
    const int max_iter = 100;
    DIIS diis;
    Eigen::MatrixXd P, C;
    Eigen::VectorXd epsilon;

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
    C = X * eig_H.eigenvectors();
    P = build_density(C, n_occ);

    // Step 3: SCF iterations
    double E_total_prev = 0.0;

    std::cout << "\nStarting SCF iterations...\n" << std::endl;

    for (int iter = 0; iter < max_iter; ++iter) {
      const Eigen::MatrixXd P_old = P; // Save for convergence check

      // Coulomb J
      Eigen::MatrixXd J = compute_coulomb(input.ERI, P);

      // XC Calculations
      auto [E_xc, V_xc] = compute_exchange_correlation(P, xc, XC_VWN);

      // Build Fock matrix
      Eigen::MatrixXd F = H_core + J + V_xc;

      // DIIS Extrapolation
      F = diis.compute(F, P, input.S);

      // Energy
      double E_one_e = compute_one_e_energy(P, H_core);
      double E_coul  = compute_coul_energy(P, J);
      double E_total = E_one_e + E_coul + E_xc + input.nuc_repulsion;

      // Diagonalize Fock
      diagonalize_fock_and_reconstruct_answer(F, X, n_occ, C, epsilon, P);

      // Convergence Check
      const double rmsd = (P - P_old).norm();
      const double deltaE = std::abs(E_total - E_total_prev);

      std::cout << "Iter " << std::setw(3) << iter + 1
                << ": E_one_e = " << std::fixed << std::setprecision(10) << E_one_e
                << ": E_coul = " << std::fixed << std::setprecision(10) << E_coul
                << ": E = " << std::fixed << std::setprecision(10) << E_total
                << ", ΔE = " << std::scientific << deltaE
                << ", RMSD = " << rmsd << std::endl;

      if (rmsd < convergence && deltaE < convergence) {
          std::cout << "\nSCF converged in " << iter + 1 << " iterations!" << std::endl;
          return {E_total, E_xc, C, epsilon, P};
      }

      E_total_prev = E_total;
  }

  throw std::runtime_error("SCF failed to converge!");
}
} // namespace SCF_LDA