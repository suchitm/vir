#ifndef __LM__
#define __LM__

#include "../typedefs.h"

// *****************************************************************************
// Variational Algorithm Helpers
// *****************************************************************************
// [[Rcpp::export]]
Rcpp::List lm_vi_b0(
  Eigen::VectorXd& y_s, Eigen::MatrixXd& X_s, Rcpp::List& param_b, 
  Rcpp::List& param_tau, int& N, int& S, Rcpp::List& param_b0
);

// [[Rcpp::export]]
Rcpp::List lm_vi_b(
  Eigen::VectorXd& y_s, Eigen::MatrixXd& X_s, Rcpp::List& param_b0,
  Rcpp::List& param_tau, Eigen::MatrixXd& mu_prior_mat, int& N, int& S, 
  int& P, int& type, bool cavi, Rcpp::List& param_b
);

#endif // __LM__
