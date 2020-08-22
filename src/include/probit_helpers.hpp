#ifndef __PROBIT__
#define __PROBIT__

#include "../typedefs.h"

// ****************************************************
// gibbs helpers
// ****************************************************
Eigen::VectorXd probit_gibbs_z(
  Eigen::VectorXi& y, Eigen::VectorXd& eta, int& N, Eigen::VectorXd& z
);

double probit_gibbs_b0(Eigen::VectorXd& ehat, int& N, double& b0);

Eigen::VectorXd probit_gibbs_b(
  Eigen::MatrixXd& X, Eigen::VectorXd& ehat, Eigen::MatrixXd& prior_mat, 
  int& P, Eigen::VectorXd& b
);

double probit_log_lik(
  Eigen::VectorXi& y, Eigen::MatrixXd& X, double& b0, Eigen::VectorXd& b, 
  int& N
);

// ****************************************************
// vi helpers
// ****************************************************

Rcpp::List probit_vi_z(
  Eigen::MatrixXd& X_s, Rcpp::List& param_b0, Rcpp::List& param_b,
  int& S, Rcpp::List& param_z
);

Rcpp::List probit_vi_b0(
  Eigen::MatrixXd& X_s, Rcpp::List& param_z, Rcpp::List& param_b, int& N,
  int& S, Rcpp::List& param_b0
);

Rcpp::List probit_vi_b(
  Eigen::MatrixXd& X_s, Rcpp::List& param_z, Rcpp::List& param_b0,
  Eigen::MatrixXd& mu_prior_mat, int& N, int& S, int& P, int& type,
  bool cavi, Rcpp::List& param_b
);

#endif // __PROBIT__
