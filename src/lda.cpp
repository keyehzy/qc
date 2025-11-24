#include "lda.h"

#include <iostream>
#include <iomanip>

namespace SCF_LDA {
Eigen::VectorXd compute_density_on_grid(const Eigen::MatrixXd& P, const XC_Grid& xc) {
    // 1. Temp = Phi * P (N_grid x N_orb)
    Eigen::MatrixXd Temp = xc.phi * P; 
    
    // 2. Row-wise dot product
    // rho(g) = dot(Temp.row(g), phi.row(g))
    return (Temp.cwiseProduct(xc.phi)).rowwise().sum();
}

struct LDA_Slater {
    static void eval(double rho, double& eps, double& v_x) {
        if (rho <= 1e-14) {
            eps = 0.0;
            v_x = 0.0;
            return;
        }
        // Cx = -3/4 * (3/pi)^(1/3)
        static const double Cx = (-3.0 / 4.0) * std::cbrt(3.0 / M_PI);
        double rho_13 = std::cbrt(rho);
        
        eps = Cx * rho_13;
        v_x = (4.0 / 3.0) * eps;
    }
};

// 2. VWN5 Correlation (Vosko, Wilk, Nusair 1980)
// Constants for Unpolarized gas
struct LDA_VWN {
    static void eval(double rho, double& eps, double& v_c) {
        if (rho <= 1e-14) {
            eps = 0.0;
            v_c = 0.0;
            return;
        }

        // Constants for Paramagnetic (Unpolarized) form
        const double A = 0.0310907;
        const double b = 3.72744;
        const double c = 12.9352;
        const double x0 = -0.10498;
        
        // Derived constants
        const double Q = std::sqrt(4.0 * c - b * b);
        const double f_Q = 2.0 * b / Q;
        const double f_x0 = b * x0 / (x0*x0 + b*x0 + c); // bx0 / X(x0)
        const double f_tan_x0 = 2.0 * (b + 2.0 * x0) / Q;

        // Wigner-Seitz radius rs
        double rs = std::cbrt(3.0 / (4.0 * M_PI * rho));
        double x  = std::sqrt(rs);
        double X  = x*x + b*x + c;

        // Parts of eps formula
        double tan_inv = std::atan(Q / (2.0 * x + b));
        double log_rat = std::log(x*x / X);
        double log_x0  = std::log((x - x0)*(x - x0) / X);

        eps = A * (log_rat + f_Q * tan_inv - f_x0 * (log_x0 + f_tan_x0 * tan_inv));

        // Derivative de/dx      
        double dx_X = (2.0*x + b);
        double term1 = 2.0/x - dx_X/X;
        double d_atan = -Q / (2.0 * X);
        double term2 = (2.0/(x-x0) - dx_X/X);
        double de_dx = A * (term1 + f_Q * d_atan - f_x0 * (term2 + f_tan_x0 * d_atan));

        // v_c = eps - (r_s / 3) * (d_eps / d_r_s)
        // r_s = x^2 => d_r_s = 2x dx
        // d_eps / d_r_s = (d_eps / dx) * (1/2x)
        // v_c = eps - (x^2 / 3) * (1/2x) * (d_eps/dx) = eps - (x/6) * de_dx
        v_c = eps - (x / 6.0) * de_dx;
    }
};

// 3. PW92 Correlation (Perdew Wang 92)
struct LDA_PW92 {
    static void eval(double rho, double& eps, double& v_c) {
        if (rho <= 1e-14) {
            eps = 0.0;
            v_c = 0.0;
            return;
        }

        const double A = 0.031091;
        const double a1 = 0.21370;
        const double b1 = 7.5957;
        const double b2 = 3.5876;
        const double b3 = 1.6382;
        const double b4 = 0.49294;

        double rs = std::cbrt(3.0 / (4.0 * M_PI * rho));
        double rs_12 = std::sqrt(rs);
        double rs_32 = rs * rs_12;
        double rs_2  = rs * rs;

        // D = 2A * (b1*rs^1/2 + b2*rs + b3*rs^3/2 + b4*rs^2)
        double denom_poly = (b1 * rs_12 + b2 * rs + b3 * rs_32 + b4 * rs_2);
        double D = 2.0 * A * denom_poly;
        
        double log_term = std::log(1.0 + 1.0 / D);
        
        eps = -2.0 * A * (1.0 + a1 * rs) * log_term;

        // Derivative for Potential
        // v_c = eps - (rs/3) * (d_eps/d_rs)
        
        double dD_drs = 2.0 * A * (0.5 * b1 / rs_12 + b2 + 1.5 * b3 * rs_12 + 2.0 * b4 * rs);
        
        // d/drs [ (1+a1 rs) * ln(1+1/D) ]
        // = a1 * ln(...) + (1+a1 rs) * (1 / (1+1/D)) * (-1/D^2) * dD_drs
        // = a1 * ln(...) - (1+a1 rs) * (D / (D+1)) * (1/D^2) * dD_drs
        // = a1 * ln(...) - (1+a1 rs) / (D * (D+1)) * dD_drs

        double deps_drs = -2.0 * A * (a1 * log_term - (1.0 + a1 * rs) * dD_drs / (D * (D + 1.0)));

        v_c = eps - (rs / 3.0) * deps_drs;
    }
};

enum CorrelationFunctional {
    XC_NONE,
    XC_VWN,
    XC_PW92
};

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

Result run_scf(const InputIntegrals& input, const XC_Grid& xc, int n_electrons) {
    
    const int norb = input.S.rows();
    const int n_occ = n_electrons / 2;
    const double convergence = 1e-6;
    const int max_iter = 50;
    const double mixing = 0.5;
    
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
    Eigen::MatrixXd P_prev = P;
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

      // XC Calculations
      Eigen::VectorXd rho_grid = compute_density_on_grid(P, xc);
      Eigen::VectorXd eps_xc_grid;
      Eigen::VectorXd v_xc_grid;
      double E_xc = eval_xc(rho_grid, xc, eps_xc_grid, v_xc_grid, XC_VWN);
      Eigen::MatrixXd V_xc = build_V_xc(xc, v_xc_grid, norb);
      
      // Build Fock matrix
      Eigen::MatrixXd F = H_core + J + V_xc;
      
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
      double E_one_e = (P.cwiseProduct(H_core)).sum();
      double E_coul  = 0.5 * (P.cwiseProduct(J)).sum();
      double E_total = E_one_e + E_coul + E_xc + input.nuc_repulsion;
      
      // Convergence
      const double rmsd = (P - P_prev).norm();
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
      P = (1.0 - mixing) * P_prev + mixing * P;
      P_prev = P;
  }
  
  throw std::runtime_error("SCF failed to converge!");
}
} // namespace SCF_LDA