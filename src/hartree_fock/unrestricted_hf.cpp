#include "unrestricted_hf.h"
#include "../diis.h"

#include <iostream>
#include <iomanip>

namespace SCF_HF {
namespace unrestricted {
Result run_scf(const InputIntegrals& input, int n_electrons, int multiplicity) {
    const int norb = input.S.rows();

    int n_alpha = (n_electrons + multiplicity - 1) / 2;
    int n_beta  = n_electrons - n_alpha;

    if (n_alpha + n_beta != n_electrons) {
      throw std::runtime_error("Invalid multiplicity for electron count");
    }

    const double convergence = 1e-8;
    const int max_iter = 100;
    DIIS diis_alpha, diis_beta;

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

    Eigen::MatrixXd C_alpha_occ = C.leftCols(n_alpha);
    Eigen::MatrixXd P_alpha = C_alpha_occ * C_alpha_occ.transpose();

    Eigen::MatrixXd C_beta_occ = C.leftCols(n_beta);
    Eigen::MatrixXd P_beta = C_beta_occ * C_beta_occ.transpose();

    // Break initial symmetry
    if (n_alpha == n_beta) {
      P_beta(0,0) += 1e-4;
    }

    // Step 3: SCF iterations
    double E_total_prev = 0.0;

    std::cout << "\nStarting SCF iterations...\n" << std::endl;

    for (int iter = 0; iter < max_iter; ++iter) {
      // Map to Tensor
      Eigen::MatrixXd P_total = P_alpha + P_beta;
      Eigen::TensorMap<Eigen::Tensor<double, 2>> P_total_tensor(P_total.data(), norb, norb);

      // Coulomb J
      // J_ij = Sum_kl (ij|kl) * P_kl
      // Contract ERI dims 2,3 with P dims 0,1
      Eigen::array<Eigen::IndexPair<int>, 2> j_contract = {
        Eigen::IndexPair<int>(2, 0),  // k
        Eigen::IndexPair<int>(3, 1)   // l
      };
      Eigen::Tensor<double, 2> J_tensor = input.ERI.contract(P_total_tensor, j_contract);
      Eigen::MatrixXd J = Eigen::Map<Eigen::MatrixXd>(J_tensor.data(), norb, norb);

      // Exchange K
      // K_ij = Sum_kl (ik|jl) * P_kl
      // Contract ERI dim 1 (k) with P dim 0, and ERI dim 3 (l) with P dim 1
      Eigen::array<Eigen::IndexPair<int>, 2> k_contract = {
          Eigen::IndexPair<int>(1, 0),  // k
          Eigen::IndexPair<int>(3, 1)   // l
      };
      Eigen::TensorMap<Eigen::Tensor<double, 2>> P_alpha_tensor(P_alpha.data(), norb, norb);
      Eigen::Tensor<double, 2> K_alpha_tensor = input.ERI.contract(P_alpha_tensor, k_contract);
      Eigen::MatrixXd K_alpha = Eigen::Map<Eigen::MatrixXd>(K_alpha_tensor.data(), norb, norb);

      Eigen::TensorMap<Eigen::Tensor<double, 2>> P_beta_tensor(P_beta.data(), norb, norb);
      Eigen::Tensor<double, 2> K_beta_tensor = input.ERI.contract(P_beta_tensor, k_contract);
      Eigen::MatrixXd K_beta = Eigen::Map<Eigen::MatrixXd>(K_beta_tensor.data(), norb, norb);

      // Build Fock matrix
      Eigen::MatrixXd F_alpha = H_core + J - K_alpha;
      Eigen::MatrixXd F_beta = H_core + J - K_beta;

      // DIIS Extrapolation
      Eigen::MatrixXd F_alpha_diis = diis_alpha.compute(F_alpha, P_alpha, input.S);
      Eigen::MatrixXd F_beta_diis = diis_beta.compute(F_beta, P_beta, input.S);

      // Energy
      const double E_alpha_elec = 0.5 * (P_alpha.cwiseProduct(H_core + F_alpha)).sum();
      const double E_beta_elec = 0.5 * (P_beta.cwiseProduct(H_core + F_beta)).sum();
      const double E_total = E_alpha_elec + E_beta_elec + input.nuc_repulsion;

      // Diagonalize Fock
      const Eigen::MatrixXd F_alpha_ortho = X.transpose() * F_alpha_diis * X;
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_F_alpha(F_alpha_ortho);
      const Eigen::MatrixXd C_alpha_ortho = eig_F_alpha.eigenvectors();
      const Eigen::VectorXd epsilon_alpha = eig_F_alpha.eigenvalues();

      const Eigen::MatrixXd F_beta_ortho = X.transpose() * F_beta_diis * X;
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_F_beta(F_beta_ortho);
      const Eigen::MatrixXd C_beta_ortho = eig_F_beta.eigenvectors();
      const Eigen::VectorXd epsilon_beta = eig_F_beta.eigenvalues();

      // New density
      Eigen::MatrixXd C_alpha_new = X * C_alpha_ortho;
      Eigen::MatrixXd C_alpha_occ_new = C_alpha_new.leftCols(n_alpha);
      Eigen::MatrixXd P_alpha_new = C_alpha_occ_new * C_alpha_occ_new.transpose();

      Eigen::MatrixXd C_beta_new = X * C_beta_ortho;
      Eigen::MatrixXd C_beta_occ_new = C_beta_new.leftCols(n_beta);
      Eigen::MatrixXd P_beta_new = C_beta_occ_new * C_beta_occ_new.transpose();

      // Convergence Check
      const double rmsd_alpha = (P_alpha_new - P_alpha).norm();
      const double rmsd_beta = (P_beta_new - P_beta).norm();
      const double deltaE = std::abs(E_total - E_total_prev);

      std::cout << "Iter " << std::setw(3) << iter + 1
                << ": E_alpha_elec = " << std::fixed << std::setprecision(10) << E_alpha_elec
                << ": E_beta_elec = " << std::fixed << std::setprecision(10) << E_beta_elec
                << ": E_elec = " << std::fixed << std::setprecision(10) << E_alpha_elec + E_beta_elec
                << ": E = " << std::fixed << std::setprecision(10) << E_total
                << ", ΔE = " << std::scientific << deltaE
                << ", RMSD alpha = " << rmsd_alpha
                << ", RMSD beta = " << rmsd_beta << std::endl;

      if (rmsd_alpha < convergence && rmsd_beta < convergence && deltaE < convergence) {
          std::cout << "\nSCF converged in " << iter + 1 << " iterations!" << std::endl;
          Result result;
          result.energy = E_total;
          result.alpha = {C_alpha_new, epsilon_alpha, P_alpha_new};
          result.beta = {C_beta_new, epsilon_beta, P_beta_new};
          return result;
      }

      E_total_prev = E_total;
      P_alpha = P_alpha_new;
      P_beta = P_beta_new;
  }

  throw std::runtime_error("SCF failed to converge!");
}
} // namespace unrestricted
} // namespace SCF_HF