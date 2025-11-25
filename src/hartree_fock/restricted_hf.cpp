#include "restricted_hf.h"
#include "../diis.h"

#include <iostream>
#include <iomanip>

namespace SCF_HF {
namespace restricted {
namespace {
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

Eigen::MatrixXd compute_exchange(const Eigen::Tensor<double, 4>& ERI, const Eigen::MatrixXd& P) {
    Eigen::TensorMap<Eigen::Tensor<double, 2>> P_map(const_cast<double*>(P.data()), P.rows(), P.cols());
    Eigen::array<Eigen::IndexPair<int>, 2> contr = {
        Eigen::IndexPair<int>(1, 0),  // k index
        Eigen::IndexPair<int>(3, 1)   // l index
    };
    Eigen::Tensor<double, 2> K_tensor = ERI.contract(P_map, contr);
    return Eigen::Map<Eigen::MatrixXd>(K_tensor.data(), P.rows(), P.cols());
}

void diagonalize_fock_and_reconstruct_answer(const Eigen::MatrixXd& F, const Eigen::MatrixXd& X, int n_occ, Eigen::MatrixXd& C, Eigen::VectorXd& eps, Eigen::MatrixXd& P) {
    const Eigen::MatrixXd F_ortho = X.transpose() * F * X;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(F_ortho);
    C = X * eig.eigenvectors();
    eps = eig.eigenvalues();
    P = build_density(C, n_occ);
}

double compute_elec_energy(const Eigen::MatrixXd& P, const Eigen::MatrixXd& H_core, const Eigen::MatrixXd& F) {
    return 0.5 * (P.cwiseProduct(H_core + F)).sum();
}
} // anonymous namespace

Result run_scf(const InputIntegrals& input, int n_electrons) {
    if (n_electrons % 2 != 0) {
      throw std::runtime_error("Invalid electron count for closed shell");
    }

    const int norb = input.S.rows();
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
    P = build_density(C, norb);

    // Step 3: SCF iterations
    double E_total_prev = 0.0;

    std::cout << "\nStarting SCF iterations...\n" << std::endl;

    for (int iter = 0; iter < max_iter; ++iter) {
      const Eigen::MatrixXd P_old = P; // Save for convergence check

      // Coulomb J
      Eigen::MatrixXd J = compute_coulomb(input.ERI, P);

      // Exchange K
      Eigen::MatrixXd K = compute_exchange(input.ERI, P);

      // Build Fock matrix
      Eigen::MatrixXd F = H_core + J - 0.5 * K;

      // DIIS Extrapolation
      F = diis.compute(F, P, input.S);

      // Energy
      const double E_elec = compute_elec_energy(P, H_core, F);
      const double E_total = E_elec + input.nuc_repulsion;

      // Diagonalize Fock
      diagonalize_fock_and_reconstruct_answer(F, X, n_occ, C, epsilon, P);

      // Convergence Check
      const double rmsd = (P - P_old).norm();
      const double deltaE = std::abs(E_total - E_total_prev);

      std::cout << "Iter " << std::setw(3) << iter + 1
                << ": E_elec = " << std::fixed << std::setprecision(10) << E_elec
                << ": E = " << std::fixed << std::setprecision(10) << E_total
                << ", ΔE = " << std::scientific << deltaE
                << ", RMSD = " << rmsd << std::endl;

      if (rmsd < convergence && deltaE < convergence) {
          std::cout << "\nSCF converged in " << iter + 1 << " iterations!" << std::endl;
          return {E_total, C, epsilon, P};
      }

      E_total_prev = E_total;
  }

  throw std::runtime_error("SCF failed to converge!");
}
} // namespace restricted
} // namespace SCF_HF