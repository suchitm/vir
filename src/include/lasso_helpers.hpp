#ifndef __LASSO__
#define __LASSO__

#include "../typedefs.h"

Rcpp::List lasso_vi_lambda2(
  Rcpp::List& param_b, Rcpp::List& param_gamma, int& P, double& a_lambda2,
  double& b_lambda2, Rcpp::List& param_lambda2
);

Rcpp::List lasso_vi_gamma(
  Rcpp::List& param_b, Rcpp::List& param_tau, Rcpp::List& param_lambda2,
  int& P, Rcpp::List& param_gamma
);

Rcpp::List lasso_vi_tau(
  Eigen::VectorXd& y_n, Eigen::MatrixXd& X_n, Rcpp::List& param_b0,
  Rcpp::List& param_b, Rcpp::List& param_gamma, int& N, int& P, int& S,
  double& a_tau, double& b_tau, Rcpp::List& param_tau
);

#endif // __LASSO__
