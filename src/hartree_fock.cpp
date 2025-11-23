#include "hartree_fock.h"

#include <iostream>
#include <iomanip>

namespace HartreeFock {
Result run_scf(const InputIntegrals& input, int n_electrons) {
    
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
    Eigen::MatrixXd P_prev = Eigen::MatrixXd::Zero(P.rows(), P.cols());
    double E_elec_prev = 0.0;
    
    std::cout << "\nStarting SCF iterations...\n" << std::endl;
    
    for (int iter = 0; iter < max_iter; ++iter) {
      P_prev.swap(P);

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
      const double E_elec = 0.5 * (P.cwiseProduct(H_core + F)).sum();
      const double E_total = E_elec + input.nuc_repulsion;
      
      // Convergence
      const double rmsd = (P - P_prev).norm();
      const double deltaE = std::abs(E_elec - E_elec_prev);
      
      std::cout << "Iter " << std::setw(3) << iter + 1 
                << ": E = " << std::fixed << std::setprecision(10) << E_total
                << ", ΔE = " << std::scientific << deltaE
                << ", RMSD = " << rmsd << std::endl;
      
      if (rmsd < convergence && deltaE < convergence) {
          std::cout << "\nSCF converged in " << iter + 1 << " iterations!" << std::endl;
          return {E_total, C, epsilon, P};
      }
      
      E_elec_prev = E_elec;
  }
  
  throw std::runtime_error("SCF failed to converge!");
}
} // namespace HartreeFock