#include <iostream>
#include <cassert>
#include <algorithm>

#include "factorial.h"
#include "hermite_aux.h"
#include "orbital.h"
#include "vec3.h"

#include "basis_set.h"
#include "basis_set/sto_3g.h"

#include <fstream>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>


// https://chemistry.montana.edu/callis/courses/chmy564/460water.pdf


std::vector<Atom> parse_molecule(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

  std::string line;
  std::getline(file, line); // Skip first line

  std::vector<Atom> atoms;

  while (std::getline(file, line)) {
    std::istringstream iss(line);
    double z;
    double v1, v2, v3;

    if (iss >> z >> v1 >> v2 >> v3) {
        atoms.push_back({(int)z, Vec3{v1,v2,v3}});
    }
  }

  return atoms;
}

struct InputIntegrals {
  Eigen::MatrixXd S;
  Eigen::MatrixXd T;
  Eigen::MatrixXd V;
  Eigen::MatrixXd mux, muy, muz;
  Eigen::Tensor<double, 4> ERI;
  double nuc_repulsion;


  InputIntegrals(const std::vector<Atom>& molecule, const std::vector<ContractedGaussianTypeOrbital>& orbitals) {
    // Overlap
    S.setZero(orbitals.size(), orbitals.size());
    for (int i = 0; i < orbitals.size(); ++i) {
      for (int j = 0; j < orbitals.size(); ++j) {
        S(i,j) = orbitals[i].overlap(orbitals[j]);
      }
    }  
    
    // Kinetic
    T.setZero(orbitals.size(), orbitals.size());
    for (int i = 0; i < orbitals.size(); ++i) {
      for (int j = 0; j < orbitals.size(); ++j) {
        T(i,j) = orbitals[i].kinetic(orbitals[j]);
      }
    }

    // Nuclear attraction
    V.setZero(orbitals.size(), orbitals.size());
    for (int i = 0; i < orbitals.size(); ++i) {
      for (int j = 0; j < orbitals.size(); ++j) {
        double result = 0;
        for (const Atom& atom : molecule) {
          result += -atom.number * orbitals[i].nuclear_attraction(orbitals[j], atom.center);
        }
        V(i,j) = result;
      }
    }

    // Dipole moments
    mux.setZero(orbitals.size(), orbitals.size());
    for (int i = 0; i < orbitals.size(); ++i) {
      for (int j = 0; j < orbitals.size(); ++j) {
        mux(i,j) = orbitals[i].multipole(orbitals[j], {0,0,0}, {1,0,0});
      }
    }

    muy.setZero(orbitals.size(), orbitals.size());
    for (int i = 0; i < orbitals.size(); ++i) {
      for (int j = 0; j < orbitals.size(); ++j) {
        muy(i,j) = orbitals[i].multipole(orbitals[j], {0,0,0}, {0,1,0});
      }
    }
    
    muz.setZero(orbitals.size(), orbitals.size());
    for (int i = 0; i < orbitals.size(); ++i) {
      for (int j = 0; j < orbitals.size(); ++j) {
        muz(i,j) = orbitals[i].multipole(orbitals[j], {0,0,0}, {0,0,1});
      }
    }
   

    // Electron-electron repulsion
    ERI = Eigen::Tensor<double, 4>((int)orbitals.size(), (int)orbitals.size(), (int)orbitals.size(), (int)orbitals.size());
    ERI.setZero();
    for (int i = 0; i < orbitals.size(); ++i) {
      for (int j = 0; j < orbitals.size(); ++j) {
        for (int k = 0; k < orbitals.size(); ++k) {
          for (int l = 0; l < orbitals.size(); ++l) {
            ERI(i, j, k, l) = ContractedGaussianTypeOrbital::electron_repulsion(orbitals[i], orbitals[j], orbitals[k], orbitals[l]);
          }
        }
      }
    }
    
    // Nuclear repulsion
    nuc_repulsion = 0.0;
    for (int i = 0; i < molecule.size(); ++i) {
      for (int j = 0; j < molecule.size(); ++j) {
        if (i < j) {
          nuc_repulsion += molecule[i].number * molecule[j].number / (molecule[i].center - molecule[j].center).norm();
        }
      }
    }  
  }
};

struct SCF_Result {
  double energy;
  Eigen::MatrixXd C;
  Eigen::VectorXd epsilon;
  Eigen::MatrixXd P;
};

SCF_Result run_scf(const InputIntegrals& input, int n_electrons) {
    
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
    
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(norb, norb);
    for (int i = 0; i < norb; ++i) {
      for (int j = 0; j < norb; ++j) {
        for (int k = 0; k < n_occ; ++k) {
            P(i, j) += 2.0 * C(i, k) * C(j, k);
        }
      }
    }
  
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
      P.setZero();
      for (int i = 0; i < norb; ++i) {
        for (int j = 0; j < norb; ++j) {
          for (int k = 0; k < n_occ; ++k) {
            P(i, j) += 2.0 * C(i, k) * C(j, k);
          }
        }
      }
      
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

int main() {
  auto molecule = parse_molecule("./assets/ch4/STO-3G/geom.dat");
  auto orbitals = convert(molecule, STO_3G);
  auto integrals = InputIntegrals(molecule, orbitals);

  int n_electrons = 10;
  auto result = run_scf(integrals, n_electrons);
  return 0;
}

