#ifndef __MULTIVARIATE__
#define __MULTIVARIATE__

#include "../typedefs.h"

// *****************************************************************************
// gibbs samplers
// *****************************************************************************
Eigen::MatrixXd mv_probit_gibbs_Z(
  Eigen::MatrixXi& Y, Eigen::MatrixXd& Eta, Eigen::VectorXd& tau, int& N, 
  int& M, Eigen::MatrixXd& Z
);

Eigen::VectorXd mvlm_uninf_gibbs_b0(
  Eigen::MatrixXd& E_hat, Eigen::VectorXd& tau, int& N, int& M, 
  Eigen::VectorXd& b0
);

Eigen::MatrixXd mvlm_uninf_gibbs_B(
  Eigen::MatrixXd& E_hat, Eigen::MatrixXd& X, Eigen::VectorXd& tau, int& M,
  int& P, Eigen::MatrixXd& b
);

Eigen::MatrixXd mvlm_uninf_gibbs_mtheta(
  Eigen::MatrixXd& E_hat, Eigen::MatrixXd& mphi, Eigen::MatrixXd& mgamma,
  Eigen::VectorXd& tau, Eigen::VectorXd& lambda, int& M, int& K,
  Eigen::MatrixXd& mtheta
);

Eigen::MatrixXd mvlm_uninf_gibbs_mphi(
  Eigen::MatrixXd& E_hat, Eigen::MatrixXd& mtheta, Eigen::VectorXd& tau,
  int& N, int& K, Eigen::MatrixXd& mphi
);

Eigen::VectorXd mvlm_uninf_gibbs_tau(
  Eigen::MatrixXd& E_hat, int& N, int& M, double& a_tau, double& b_tau,
  Eigen::VectorXd& tau
);

// [[Rcpp::export]]
Eigen::MatrixXd mvlm_uninf_gibbs_mgamma(
  Eigen::MatrixXd& mtheta, Eigen::VectorXd& lambda, int& M, int& K,
  double& nu, Eigen::MatrixXd& mgamma
);

// [[Rcpp::export]]
Eigen::VectorXd mvlm_uninf_gibbs_xi(
  Eigen::MatrixXd& mtheta, Eigen::MatrixXd& mgamma, Eigen::VectorXd& lambda,
  int& M, int& K, double& a1, double& a2, Eigen::VectorXd& xi
);

// *****************************************************************************
// variational algorithms
// *****************************************************************************
Rcpp::List mvlm_vi_phi(
  Eigen::MatrixXd& E_hat, Rcpp::List& param_theta, Rcpp::List& param_tau,
  int& S, int& M, int& K, Rcpp::List& param_phi
);

Rcpp::List mvlm_vi_theta(
  Eigen::MatrixXd& E_hat, Rcpp::List& param_phi, Rcpp::List& param_tau,
  Rcpp::List& param_gamma, Eigen::VectorXd& mu_lambda, int& N, int& M,
  int& S, int& K, Rcpp::List& param_theta
);

Rcpp::List mvlm_vi_b0(
  Eigen::MatrixXd& E_hat, Rcpp::List& param_tau, int& N, int& M, int& S,
  Rcpp::List& param_b0
);

Rcpp::List mvlm_vi_b(
  Eigen::MatrixXd& E_hat, Eigen::MatrixXd& X, Rcpp::List& param_tau,
  int& N, int& M, int& P, int& S, Rcpp::List& param_b
);

Rcpp::List mvlm_vi_tau (
  Eigen::MatrixXd& E_hat, Eigen::MatrixXd& X, Rcpp::List& param_b0,
  Rcpp::List& param_b, Rcpp::List& param_phi, Rcpp::List& param_theta,
  int& N, int& M, int& P, int& S, int& K, double& a_tau, double& b_tau,
  Rcpp::List& param_tau
);

Rcpp::List mvlm_vi_gamma(
  Rcpp::List& param_theta, Eigen::VectorXd& mu_lambda, int& M, int& K,
  double& nu, Rcpp::List& param_gamma
);

Rcpp::List mvlm_vi_xi(
  Rcpp::List& param_gamma, Rcpp::List& param_theta, int& M, int& K,
  Eigen::VectorXd& mu_lambda, double& a1, double& a2, Rcpp::List& param_xi,
  bool svb = false
);

// *****************************************************************************
// others
// *****************************************************************************
Eigen::VectorXd cum_prod(Eigen::VectorXd& x);

#endif // __MULTIVARIATE__
