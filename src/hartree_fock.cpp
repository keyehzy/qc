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
    Eigen::MatrixXd P_prev;
    double E_elec_prev = 0.0;
    
    std::cout << "\nStarting SCF iterations...\n" << std::endl;
    
    for (int iter = 0; iter < max_iter; ++iter) {
        P_prev = P;
        
      // Build Fock matrix
      Eigen::MatrixXd F = Eigen::MatrixXd::Zero(norb, norb);
      for (int i = 0; i < norb; ++i) {
        for (int j = 0; j <= i; ++j) {
          double G = 0.0;
          for (int k = 0; k < norb; ++k) {
            for (int l = 0; l < norb; ++l) {
              const double P_kl = P(k, l);
              const double coulomb = input.ERI(i, j, k, l);
              const double exchange = input.ERI(i, k, j, l);
              G += P_kl * (coulomb - 0.5 * exchange);
            }
          }
          F(i, j) = H_core(i, j) + G;
          F(j, i) = F(i, j);
        }
      }
      
      // Diagonalize Fock
      const Eigen::MatrixXd F_ortho = X.transpose() * F * X;
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_F(F_ortho);
      const Eigen::MatrixXd C_ortho = eig_F.eigenvectors();
      const Eigen::VectorXd epsilon = eig_F.eigenvalues();
      C = X * C_ortho;
      
      // New density
      auto C_occ = C.leftCols(n_occ);
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