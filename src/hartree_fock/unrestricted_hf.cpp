#include "unrestricted_hf.h"
#include "../diis.h"

#include <iostream>
#include <iomanip>

namespace SCF_HF {
namespace unrestricted {
namespace {
Eigen::MatrixXd build_density(const Eigen::MatrixXd& C, int n_occ) {
    return C.leftCols(n_occ) * C.leftCols(n_occ).transpose();
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
    Eigen::MatrixXd K = Eigen::Map<Eigen::MatrixXd>(K_tensor.data(), P.rows(), P.cols());
    return K;
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

Result run_scf(const InputIntegrals& input, int n_electrons, int multiplicity) {
    const int n_alpha = (n_electrons + multiplicity - 1) / 2;
    const int n_beta  = n_electrons - n_alpha;

    if (n_alpha + n_beta != n_electrons) {
      throw std::runtime_error("Invalid multiplicity for electron count");
    }

    const double convergence = 1e-8;
    const int max_iter = 100;

    // Spin channel data: 0=alpha, 1=beta
    std::array<Eigen::MatrixXd, 2> P, C;
    std::array<Eigen::VectorXd, 2> epsilon;
    std::array<DIIS, 2> diis;
    const std::array<int, 2> n_occ = {n_alpha, n_beta};

    // Step 1: Core Hamiltonian & Orthogonalizer
    const Eigen::MatrixXd H_core = input.T + input.V;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_S(input.S);
    if (eig_S.eigenvalues().minCoeff() < 1e-10) {
        throw std::runtime_error("Overlap matrix is near-singular!");
    }

    const Eigen::MatrixXd X = eig_S.eigenvectors() * eig_S.eigenvalues().cwiseSqrt().cwiseInverse().asDiagonal() * eig_S.eigenvectors().transpose();

    // Step 2: Initial guess from H_core
    const Eigen::MatrixXd H_ortho = X.transpose() * H_core * X;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_H(H_ortho);
    C[0] = C[1] = X * eig_H.eigenvectors();
    epsilon[0] = epsilon[1] = eig_H.eigenvalues();

    P[0] = build_density(C[0], n_alpha);
    P[1] = build_density(C[1], n_beta);

    // Break symmetry for open-shell cases
    if (n_alpha == n_beta) {
        P[1](0,0) += 1e-4;
    }

    double E_total_prev = 0.0;
    std::cout << "\nStarting SCF iterations...\n" << std::endl;

    for (int iter = 0; iter < max_iter; ++iter) {
        const std::array<Eigen::MatrixXd, 2> P_old = P; // Save for convergence check
        const Eigen::MatrixXd P_total = P[0] + P[1];

        // Compute Coulomb J (common to both spins)
        Eigen::MatrixXd J = compute_coulomb(input.ERI, P_total);

        // Process spin channels
        std::array<Eigen::MatrixXd, 2> F;
        std::array<double, 2> E_spin;

        for (int s = 0; s < 2; ++s) {
            // Build Fock: F = H_core + J - K
            F[s] = H_core + J - compute_exchange(input.ERI, P[s]);

            // DIIS extrapolation
            F[s] = diis[s].compute(F[s], P[s], input.S);

            // Energy for this spin
            E_spin[s] = compute_elec_energy(P[s], H_core, F[s]);
        }

        const double E_elec = E_spin[0] + E_spin[1];
        const double E_total = E_elec + input.nuc_repulsion;

        // Diagonalize and form new densities
        for (int s = 0; s < 2; ++s) {
            diagonalize_fock_and_reconstruct_answer(F[s], X, n_occ[s], C[s], epsilon[s], P[s]);
        }

        // Convergence check
        const double deltaE = std::abs(E_total - E_total_prev);
        const double rmsd_alpha = (P[0] - P_old[0]).norm();
        const double rmsd_beta = (P[1] - P_old[1]).norm();

        std::cout << "Iter " << std::setw(3) << iter + 1
                  << ": E_alpha_elec = " << std::fixed << std::setprecision(10) << E_spin[0]
                  << ": E_beta_elec = " << std::fixed << std::setprecision(10) << E_spin[1]
                  << ": E_elec = " << std::fixed << std::setprecision(10) << E_elec
                  << ": E = " << std::fixed << std::setprecision(10) << E_total
                  << ", ΔE = " << std::scientific << deltaE
                  << ", RMSD alpha = " << rmsd_alpha
                  << ", RMSD beta = " << rmsd_beta << std::endl;

        if (rmsd_alpha < convergence && rmsd_beta < convergence && deltaE < convergence) {
            std::cout << "\nSCF converged in " << iter + 1 << " iterations!" << std::endl;
            return {E_total, {C[0], epsilon[0], P[0]}, {C[1], epsilon[1], P[1]}};
        }

        E_total_prev = E_total;
    }

    throw std::runtime_error("SCF failed to converge!");
}
} // namespace unrestricted
} // namespace SCF_HF