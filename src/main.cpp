#include <iostream>
#include <cassert>
#include <algorithm>

#include "factorial.h"
#include "hermite_aux.h"
#include "orbital.h"
#include "vec3.h"

#include "basis_set.h"
#include "basis_set/sto_3g_simplified.h"

#include <fstream>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>


// https://chemistry.montana.edu/callis/courses/chmy564/460water.pdf

std::vector<GaussianTypeOrbital::Exponent> find_exponents(int N) {
  std::vector<GaussianTypeOrbital::Exponent> result;
  result.reserve((N + 1) * (N + 2) / 2);
  for (int i = 0; i <= N; ++i) {
    for (int j = 0; j <= N - i; ++j) {
      int k = N - i - j;
      result.push_back({i, j, k});
    }
  }
  std::reverse(result.begin(), result.end());
  return result;
}

std::vector<ContractedGaussianTypeOrbital> cgto_from_basis_set(const std::vector<Atom>& molecule, const BasisSet& basis_set) {
  std::vector<ContractedGaussianTypeOrbital> result;
  for(const auto& atom : molecule) {
    for (const auto& shell : basis_set.at(atom.number)) {
      for (int a : shell.angular_momentum) {
        for (const auto& exponent : find_exponents(a)) {
          std::vector<ContractedGaussianTypeOrbital::Param> params;
          for (size_t j = 0; j < shell.alphas.size(); j++) {
            params.push_back({shell.coefficients[a][j], shell.alphas[j]});
          }
          result.push_back({atom.center, exponent, params});
        }
      }
    }
  }
  return result;
}

std::vector<Atom> parse_geometry(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

  std::string line;
  std::getline(file, line); // Skip first line

  std::vector<Atom> atoms;

  while (std::getline(file, line)) {
    std::istringstream iss(line);
    int z;
    double v1, v2, v3;

    if (iss >> z >> v1 >> v2 >> v3) {
        atoms.push_back({z, Vec3{v1,v2,v3}});
    }
  }

  return atoms;
}


Eigen::MatrixXd parse_matrix(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    using Triplet = std::tuple<int, int, double>;

    std::vector<Triplet> triplets;
    std::string line;
    int max_row = 0;
    int max_col = 0;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int i, j;
        double value;

        if (iss >> i >> j >> value) {
            i--;
            j--;

            triplets.emplace_back(i, j, value);
            max_row = std::max(max_row, i);
            max_col = std::max(max_col, j);
        }
    }

    int rows = max_row + 1;
    int cols = max_col + 1;

    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(rows, cols);
    for (const auto& triplet : triplets) {
        auto [row, col, value] = triplet;
        matrix(row, col) = value;
    }

    return matrix;
}

Eigen::Tensor<double, 4> parse_matrix_5(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    using Tuple = std::tuple<int, int, int, int, double>;

    std::vector<Tuple> tuples;
    std::string line;
    int max_i = 0;
    int max_j = 0;
    int max_k = 0;
    int max_l = 0;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int i, j, k, l;
        double value;

        if (iss >> i >> j >> k >> l >> value) {
            i--;
            j--;
            k--;
            l--;

            tuples.emplace_back(i, j, k, l, value);
            max_i = std::max(max_i, i);
            max_j = std::max(max_j, j);
            max_k = std::max(max_k, k);
            max_l = std::max(max_l, l);
        }
    }

    int is = max_i + 1;
    int js = max_j + 1;
    int ks = max_k + 1;
    int ls = max_l + 1;

    Eigen::Tensor<double, 4> tensor(is, js, ks, ls);
    tensor.setZero();
    for (const auto& tuple : tuples) {
        auto [i, j, k, l, value] = tuple;
        tensor(i, j, k, l) = value;
    }

    return tensor;
}

int main() {
  auto H2O = parse_geometry("./assets/h2o/STO-3G/geom.dat");
  auto orbitals = cgto_from_basis_set(H2O, STO_3G);

  auto bench_S = parse_matrix("./assets/h2o/STO-3G/s.dat");
  std::cout << bench_S << "\n\n";

  Eigen::MatrixXd S = Eigen::MatrixXd::Zero(orbitals.size(), orbitals.size());
  for (int i = 0; const auto& orbital1 : orbitals) {
    for (int j = 0; const auto& orbital2 : orbitals) {
        if (j <= i) {
          S(i,j) = orbital1.overlap(orbital2);
        }
        ++j;
    }
    ++i;
  }

  std::cout << S << "\n\n";


  auto bench_T = parse_matrix("./assets/h2o/STO-3G/t.dat");
  std::cout << bench_T << "\n\n";

  Eigen::MatrixXd T = Eigen::MatrixXd::Zero(orbitals.size(), orbitals.size());
  for (int i = 0; const auto& orbital1 : orbitals) {
    for (int j = 0; const auto& orbital2 : orbitals) {
      if (j <= i) {
        T(i,j) = orbital1.kinetic(orbital2);
      }
      ++j;
    }
    ++i;
  }

  std::cout << T << "\n\n";

  auto bench_V = parse_matrix("./assets/h2o/STO-3G/v.dat");
  std::cout << bench_V << "\n\n";

  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(orbitals.size(), orbitals.size());
  for (int i = 0; const auto& orbital1 : orbitals) {
    for (int j = 0; const auto& orbital2 : orbitals) {
      if (j <= i) {
        double result = 0;
        for (const Atom& atom : H2O) {
          result += -atom.number * orbital1.nuclear_attraction(orbital2, atom.center);
        }
        V(i,j) = result;
      }
      ++j;
    }
    ++i;
  }

  std::cout << V << "\n\n";

  auto bench_ERI = parse_matrix_5("./assets/h2o/STO-3G/eri.dat");

  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 7; j++) {
      for (int k = 0; k < 7; k++) {
        for (int l = 0; l < 7; l++) {
          double value = bench_ERI(i,j,k,l);
          if (std::abs(value) < 1e-6) continue;
          std::printf("%d %d %d %d %f\n", i, j, k, l, value);
        }
      }
    }
  }
  std::cout << "\n\n";

  for (int i = 0; const auto& orbital1 : orbitals) {
    for (int j = 0; const auto& orbital2 : orbitals) {
      for (int k = 0; const auto& orbital3 : orbitals) {
        for (int l = 0; const auto& orbital4 : orbitals) {
          double value = ContractedGaussianTypeOrbital::electron_repulsion(orbital1, orbital2, orbital3, orbital4);
          if (std::abs(value) < 1e-6) continue;
          std::printf("%d %d %d %d %f\n", i, j, k, l, value);
          ++l;
        }
        ++k;
      }
      ++j;
    }
    ++i;
  }

  return 0;
}

