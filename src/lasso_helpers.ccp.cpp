#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include "typedefs.h"
#include "include/helpers.hpp"

using namespace Eigen;
using Eigen::Map;

//**********************************************************************//
// vi helpers
//**********************************************************************//
Rcpp::List lasso_vi_lambda2(
  Rcpp::List& param_b, Rcpp::List& param_gamma, int& P, double& a_lambda2,
  double& b_lambda2, Rcpp::List& param_lambda2
){
  Eigen::VectorXd mu_inv_gamma = param_gamma["mu_inv"];
  double delta1_t = P + a_lambda2 - 1;
  double delta2_t = -b_lambda2 - 1.0 / 2.0 * mu_inv_gamma.sum();

  param_lambda2["delta1_t"] = delta1_t;
  param_lambda2["delta2_t"] = delta2_t;

  return(param_lambda2);
}

Rcpp::List lasso_vi_gamma(
  Rcpp::List& param_b, Rcpp::List& param_tau, Rcpp::List& param_lambda2,
  int& P, Rcpp::List& param_gamma
){
  Eigen::VectorXd mu_b = param_b["mu"];
  Eigen::VectorXd vsigma2_b = param_b["vsigma2"];
  double mu_tau = param_tau["mu"];
  double mu_lambda2 = param_lambda2["mu"];
  Eigen::VectorXd delta1_t = param_gamma["delta1_t"];
  Eigen::VectorXd delta2_t = param_gamma["delta2_t"];

  delta1_t = -mu_tau / 2.0 * (
    mu_b.array().square() + vsigma2_b.array()
  );
  delta2_t = -mu_lambda2 / 2.0 * Eigen::VectorXd::Constant(P, 1.0);

  param_gamma["delta1_t"] = delta1_t;
  param_gamma["delta2_t"] = delta2_t;
  return(param_gamma);
}

// lm specific
Rcpp::List lasso_vi_tau(
  Eigen::VectorXd& y_n, Eigen::MatrixXd& X_n, Rcpp::List& param_b0,
  Rcpp::List& param_b, Rcpp::List& param_gamma, int& N, int& P, int& S,
  double& a_tau, double& b_tau, Rcpp::List& param_tau
){
  double mu_b0 = param_b0["mu"];
  double sigma2_b0 = param_b0["sigma2"];
  Eigen::VectorXd mu_b = param_b["mu"];
  Eigen::MatrixXd msigma_b = param_b["msigma"];
  Eigen::VectorXd mu_gamma = param_gamma["mu"];
  double delta1_t = param_tau["delta1_t"];
  double delta2_t = param_tau["delta2_t"];

  delta1_t = (N + P) / 2.0 + a_tau - 1.0;
  delta2_t = -b_tau - 1.0 / 2.0 * (
    N * 1.0 / S * (
      (y_n - Eigen::VectorXd::Ones(S, 1) * mu_b0 - X_n * mu_b).squaredNorm() +
      S * sigma2_b0 +
      (X_n.transpose() * X_n * msigma_b).trace()
    ) + (
      mu_gamma.array() * (mu_b.array().square() + msigma_b.diagonal().array())
    ).sum()
  );

  param_tau["delta1_t"] = delta1_t;
  param_tau["delta2_t"] = delta2_t;
  return(param_tau);
}
