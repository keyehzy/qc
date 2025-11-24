#pragma once

#include "orbital.h"
#include "basis_set.h"

#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

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
#pragma omp parallel for
    for (int i = 0; i < orbitals.size(); ++i) {
      for (int j = 0; j < orbitals.size(); ++j) {
        S(i,j) = orbitals[i].overlap(orbitals[j]);
      }
    }  
    
    // Kinetic
    T.setZero(orbitals.size(), orbitals.size());
#pragma omp parallel for
    for (int i = 0; i < orbitals.size(); ++i) {
      for (int j = 0; j < orbitals.size(); ++j) {
        T(i,j) = orbitals[i].kinetic(orbitals[j]);
      }
    }

    // Nuclear attraction
    V.setZero(orbitals.size(), orbitals.size());
#pragma omp parallel for
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
#pragma omp parallel for
    for (int i = 0; i < orbitals.size(); ++i) {
      for (int j = 0; j < orbitals.size(); ++j) {
        mux(i,j) = orbitals[i].multipole(orbitals[j], {0,0,0}, {1,0,0});
      }
    }

    muy.setZero(orbitals.size(), orbitals.size());
#pragma omp parallel for
    for (int i = 0; i < orbitals.size(); ++i) {
      for (int j = 0; j < orbitals.size(); ++j) {
        muy(i,j) = orbitals[i].multipole(orbitals[j], {0,0,0}, {0,1,0});
      }
    }
    
    muz.setZero(orbitals.size(), orbitals.size());
#pragma omp parallel for
    for (int i = 0; i < orbitals.size(); ++i) {
      for (int j = 0; j < orbitals.size(); ++j) {
        muz(i,j) = orbitals[i].multipole(orbitals[j], {0,0,0}, {0,0,1});
      }
    }
   

    // Electron-electron repulsion
    ERI = Eigen::Tensor<double, 4>((int)orbitals.size(), (int)orbitals.size(), (int)orbitals.size(), (int)orbitals.size());
    ERI.setZero();
#pragma omp parallel for
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
#pragma omp parallel for reduction(+:nuc_repulsion)
    for (int i = 0; i < molecule.size(); ++i) {
      for (int j = 0; j < i; ++j) {
        nuc_repulsion += molecule[i].number * molecule[j].number / (molecule[i].center - molecule[j].center).norm();
      }
    }  



}
};
