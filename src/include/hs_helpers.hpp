#ifndef __HS__
#define __HS__

#include "../typedefs.h"

Rcpp::List hs_vi_lambda(
  Rcpp::List& param_b, Rcpp::List& param_tau, Rcpp::List& param_gamma,
  Rcpp::List& param_xi, int& P, Rcpp::List& param_lambda
);

Rcpp::List hs_vi_xi(Rcpp::List& param_lambda, Rcpp::List& param_xi);

Rcpp::List hs_vi_gamma(
  Rcpp::List& param_b, Rcpp::List& param_tau, Rcpp::List& param_lambda,
  Rcpp::List& param_nu, int& P, Rcpp::List& param_gamma
);

Rcpp::List hs_vi_nu(
  Rcpp::List& param_gamma, int& P, Rcpp::List& param_nu
);

Rcpp::List hs_vi_tau(
  Eigen::VectorXd& y_n, Eigen::MatrixXd& X_n, Rcpp::List& param_b0,
  Rcpp::List& param_b, Rcpp::List& param_lambda, Rcpp::List& param_gamma,
  int& N, int& P, int& S, double& a_tau, double& b_tau, Rcpp::List& param_tau
);

#endif // __HS__
