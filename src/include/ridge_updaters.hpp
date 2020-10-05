#ifndef __RIDGE__
#define __RIDGE__

#include "../typedefs.h"

//****************************************************************************
// Samplers for normal linear model
//****************************************************************************

// variational bayes parameter update functions

Rcpp::List lm_ridge_vb_b0(
    double& y_bar, double& mu_tau, double& mu_b0, double& sigma2_b0, int& N
);

Rcpp::List lm_ridge_vb_b_corr(
    Eigen::MatrixXd& X, Eigen::VectorXd& y_tilde, Eigen::VectorXd& vmu_b,
    Eigen::MatrixXd& msigma_b, double& logdet_msigma_b, double& mu_tau,
    double& mu_lambda, int& P
);

Rcpp::List lm_ridge_vb_b_indep(
    Eigen::MatrixXd& X, Eigen::VectorXd& y_tilde, double& mu_tau,
    double& mu_lambda, int& N, int& P, Eigen::VectorXd& ehat_p,
    Eigen::VectorXd& vmu_b, Eigen::VectorXd& vsigma2_b
);

Rcpp::List lm_ridge_vb_tau(
    Eigen::MatrixXd& X, Eigen::VectorXd& y, double& mu_b0, double& sigma2_b0,
    Eigen::VectorXd& vmu_b, Eigen::MatrixXd& msigma_b, double& mu_lambda,
    double& astar_tau, double& bstar_tau, double& a_tau, double& b_tau,
    int& N, int& P
);

Rcpp::List lm_ridge_vb_lambda(
    Eigen::VectorXd& vmu_b, Eigen::MatrixXd& msigma_b, double& mu_tau,
    double& astar_lambda, double& bstar_lambda, double& a_lambda,
    double& b_lambda, int& P
);

double lm_ridge_vb_elbo(
    Eigen::MatrixXd& X, Eigen::VectorXd& y, double& mu_b0, double& sigma2_b0,
    Eigen::VectorXd& vmu_b, Eigen::MatrixXd& msigma_b, double& logdet_msigma_b,
    double& astar_lambda, double& bstar_lambda, double& astar_tau,
    double& bstar_tau, double& a_lambda, double& b_lambda, double& a_tau,
    double& b_tau, int& N, int& P
);

// gibbs sampling parameter update functions
double lm_ridge_gibbs_b0(double& y_bar, double& tau, double& b0, int& N);

Eigen::VectorXd lm_ridge_gibbs_b(
    Eigen::MatrixXd& X, Eigen::VectorXd& y, double& lambda, double& tau,
    Eigen::VectorXd& b, int& P
);

double lm_ridge_gibbs_tau(
    Eigen::VectorXd& ehat, Eigen::VectorXd& b, double& lambda,
    double& tau, double& a_tau, double& b_tau, int& N, int& P
);

double lm_ridge_gibbs_lambda(
    Eigen::VectorXd& b, double&tau, double& lambda, double& a_lambda,
    double& b_lambda, int& P
);

//****************************************************************************
// Samplers for probit regression
//****************************************************************************

// variational bayes parameter update functions
Rcpp::List get_phis(
    Eigen::VectorXd& eta, int& N, Eigen::VectorXd& phi, Eigen::VectorXd& Phi
);

Eigen::VectorXd probit_ridge_vb_z(
    Eigen::MatrixXd& X, Eigen::VectorXi& y, Eigen::VectorXd& eta,
    Eigen::VectorXd& phi, Eigen::VectorXd& Phi, int& N, Eigen::VectorXd& vmu_z
);

Rcpp::List probit_ridge_vb_b0(
    double& z_bar, int& N, double& mu_b0, double& sigma2_b0
);

Rcpp::List probit_ridge_vb_b_indep(
    Eigen::MatrixXd& X, Eigen::VectorXd& vmu_ztilde, double& mu_lambda, int& N, 
    int& P, Eigen::VectorXd& ehat_p, Eigen::VectorXd& vmu_b, 
    Eigen::VectorXd& vsigma2_b
);

Rcpp::List probit_ridge_vb_b_corr(
    Eigen::MatrixXd& X, Eigen::VectorXd& vmu_ztilde, double& logdet_msigma_b,
    double& mu_lambda, int& P, Eigen::VectorXd& vmu_b, Eigen::MatrixXd& msigma_b
);

Rcpp::List probit_ridge_vb_lambda(
    Eigen::VectorXd& vmu_b, Eigen::MatrixXd& msigma_b, double& a_lambda,
    double& b_lambda, int& P, double& astar_lambda, double& bstar_lambda
);

double probit_ridge_vb_elbo(
    Eigen::MatrixXd& X, Eigen::VectorXi& y, Eigen::VectorXd& vmu_z,
    double& mu_b0, double& sigma2_b0, Eigen::VectorXd& vmu_b,
    Eigen::MatrixXd& msigma_b, double& logdet_msigma_b, Eigen::VectorXd& eta,
    Eigen::VectorXd& phi, Eigen::VectorXd& Phi, double& astar_lambda,
    double& bstar_lambda, double& a_lambda, double& b_lambda, int& N, int& P
);

#endif // __RIDGE__
