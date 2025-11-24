#pragma once

namespace SCF_LDA {
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
} // namespace SCF_LDA