#pragma once

#include <vector>
#include <deque>
#include <eigen3/Eigen/Dense>

class DIIS {
private:
    std::deque<Eigen::MatrixXd> f_store; // Store previous Fock matrices
    std::deque<Eigen::MatrixXd> e_store; // Store previous Error vectors
    int max_history;

public:
    DIIS(int history_size = 6) : max_history(history_size) {}

    // Compute the DIIS extrapolated Fock matrix
    Eigen::MatrixXd compute(const Eigen::MatrixXd& F, const Eigen::MatrixXd& P, const Eigen::MatrixXd& S) {
        
        // 1. Calculate Error Vector: e = FPS - SPF (Pulay Error)
        // This vector is 0 when F and P commute (convergence)
        Eigen::MatrixXd FPS = F * P * S;
        Eigen::MatrixXd err = FPS - FPS.transpose();

        // 2. Update History
        f_store.push_back(F);
        e_store.push_back(err);
        
        if (f_store.size() > max_history) {
            f_store.pop_front();
            e_store.pop_front();
        }

        int n_dim = f_store.size();
        
        // If we don't have enough history, just return current F
        if (n_dim < 2) return F;

        // 3. Build the Pulay B Matrix
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n_dim + 1, n_dim + 1);
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n_dim + 1);
        rhs(n_dim) = -1.0;

        for (int i = 0; i < n_dim; ++i) {
            for (int j = 0; j <= i; ++j) {
                // Dot product of error matrices
                double dot = (e_store[i].cwiseProduct(e_store[j])).sum();
                B(i, j) = dot;
                B(j, i) = dot;
            }
            B(i, n_dim) = -1.0;
            B(n_dim, i) = -1.0;
        }

        // 4. Solve for Coefficients c
        // We use ColPivHouseholderQr for stability with potentially singular B
        Eigen::VectorXd c = B.colPivHouseholderQr().solve(rhs);

        // 5. Construct Extrapolated Fock Matrix
        // F_new = Sum(c_i * F_i)
        Eigen::MatrixXd F_diis = Eigen::MatrixXd::Zero(F.rows(), F.cols());
        for (int i = 0; i < n_dim; ++i) {
            F_diis += c(i) * f_store[i];
        }

        return F_diis;
    }
    
    void reset() {
        f_store.clear();
        e_store.clear();
    }
};
