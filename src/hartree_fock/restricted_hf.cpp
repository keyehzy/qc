#include "restricted_hf.h"
#include "../diis.h"

#include <iostream>
#include <iomanip>

namespace SCF_HF {
namespace restricted {
Result run_scf(const InputIntegrals& input, int n_electrons) {
    if (n_electrons % 2 != 0) {
      throw std::runtime_error("Invalid electron count for closed shell");
    }

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

      // Exchange K
      // K_ij = Sum_kl (ik|jl) * P_kl
      // Contract ERI dim 1 (k) with P dim 0, and ERI dim 3 (l) with P dim 1
      Eigen::array<Eigen::IndexPair<int>, 2> k_contract = {
          Eigen::IndexPair<int>(1, 0),  // k
          Eigen::IndexPair<int>(3, 1)   // l
      };
      Eigen::Tensor<double, 2> K_tensor = input.ERI.contract(P_tensor, k_contract);
      Eigen::MatrixXd K = Eigen::Map<Eigen::MatrixXd>(K_tensor.data(), norb, norb);

      // Build Fock matrix
      Eigen::MatrixXd F = H_core + J - 0.5 * K;

      // DIIS Extrapolation
      Eigen::MatrixXd F_diis = diis.compute(F, P, input.S);

      // Energy
      const double E_elec = 0.5 * (P.cwiseProduct(H_core + F)).sum();
      const double E_total = E_elec + input.nuc_repulsion;

      // Diagonalize Fock
      const Eigen::MatrixXd F_ortho = X.transpose() * F_diis * X;
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_F(F_ortho);
      const Eigen::MatrixXd C_ortho = eig_F.eigenvectors();
      const Eigen::VectorXd epsilon = eig_F.eigenvalues();

      // New density
      Eigen::MatrixXd C_new = X * C_ortho;
      Eigen::MatrixXd C_occ_new = C_new.leftCols(n_occ);
      Eigen::MatrixXd P_new = 2.0 * C_occ_new * C_occ_new.transpose();

      // Convergence Check
      const double rmsd = (P_new - P).norm();
      const double deltaE = std::abs(E_total - E_total_prev);

      std::cout << "Iter " << std::setw(3) << iter + 1
                << ": E_elec = " << std::fixed << std::setprecision(10) << E_elec
                << ": E = " << std::fixed << std::setprecision(10) << E_total
                << ", ΔE = " << std::scientific << deltaE
                << ", RMSD = " << rmsd << std::endl;

      if (rmsd < convergence && deltaE < convergence) {
          std::cout << "\nSCF converged in " << iter + 1 << " iterations!" << std::endl;
          return {E_total, C_new, epsilon, P_new};
      }

      E_total_prev = E_total;
      P = P_new;
  }

  throw std::runtime_error("SCF failed to converge!");
}
} // namespace restricted
} // namespace SCF_HF