#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include "typedefs.h"
#include "include/helpers.hpp"

using namespace Eigen;
using Eigen::Map;

//**********************************************************************//
// vi helpers
//**********************************************************************//
Rcpp::List hs_vi_lambda(
  Rcpp::List& param_b, Rcpp::List& param_tau, Rcpp::List& param_gamma,
  Rcpp::List& param_xi, int& P, Rcpp::List& param_lambda
){

  double mu_tau = param_tau["mu"];
  double mu_xi = param_xi["mu"];
  Eigen::VectorXd mu_gamma = param_gamma["mu"];
  Eigen::VectorXd mu_b = param_b["mu"];
  Eigen::VectorXd vsigma2_b = param_b["vsigma2"];
  double delta1_t = param_lambda["delta1_t"];
  double delta2_t = param_lambda["delta2_t"];

  delta1_t = (P + 1) / 2.0 - 1.0;
  delta2_t = -mu_xi -
    mu_tau / 2.0 * (
      mu_gamma.array() * (mu_b.array().square() + vsigma2_b.array())
    ).sum();

  param_lambda["delta1_t"] = delta1_t;
  param_lambda["delta2_t"] = delta2_t;
  return(param_lambda);
}

Rcpp::List hs_vi_xi(Rcpp::List& param_lambda, Rcpp::List& param_xi)
{
  double mu_lambda = param_lambda["mu"];
  double delta1_t = 0.0;
  double delta2_t = -1.0 - mu_lambda;

  param_xi["delta1_t"] = delta1_t;
  param_xi["delta2_t"] = delta2_t;
  return(param_xi);
}

Rcpp::List hs_vi_gamma(
  Rcpp::List& param_b, Rcpp::List& param_tau, Rcpp::List& param_lambda,
  Rcpp::List& param_nu, int& P, Rcpp::List& param_gamma
){
  double mu_lambda = param_lambda["mu"];
  double mu_tau = param_tau["mu"];
  Eigen::VectorXd mu_nu = param_nu["mu"];
  Eigen::VectorXd mu_b = param_b["mu"];
  Eigen::VectorXd vsigma2_b = param_b["vsigma2"];

  Eigen::VectorXd delta1_t = param_gamma["delta1_t"];
  Eigen::VectorXd delta2_t = param_gamma["delta2_t"];

  delta1_t = Eigen::VectorXd::Constant(P, 0.0);
  delta2_t = -mu_nu.array() - mu_tau * mu_lambda / 2.0 * (
    mu_b.array().square() + vsigma2_b.array()
  );

  param_gamma["delta1_t"] = delta1_t;
  param_gamma["delta2_t"] = delta2_t;
  return(param_gamma);
}

Rcpp::List hs_vi_nu(
  Rcpp::List& param_gamma, int& P, Rcpp::List& param_nu
){
  Eigen::VectorXd mu_gamma = param_gamma["mu"];
  Eigen::VectorXd delta1_t = param_nu["delta1_t"];
  Eigen::VectorXd delta2_t = param_nu["delta2_t"];

  delta1_t = Eigen::VectorXd::Constant(P, 0.0);
  delta2_t = -Eigen::VectorXd::Constant(P, 1.0) - mu_gamma;

  param_nu["delta1_t"] = delta1_t;
  param_nu["delta2_t"] = delta2_t;
  return(param_nu);
}

//**********************************************************************//
// lm specific
//**********************************************************************//
Rcpp::List hs_vi_tau(
  Eigen::VectorXd& y_n, Eigen::MatrixXd& X_n, Rcpp::List& param_b0,
  Rcpp::List& param_b, Rcpp::List& param_lambda, Rcpp::List& param_gamma,
  int& N, int& P, int& S, double& a_tau, double& b_tau, Rcpp::List& param_tau
){
  double mu_b0 = param_b0["mu"];
  double sigma2_b0 = param_b0["sigma2"];
  Eigen::VectorXd mu_b = param_b["mu"];
  Eigen::MatrixXd msigma_b = param_b["msigma"];
  Eigen::VectorXd mu_gamma = param_gamma["mu"];
  double mu_lambda = param_lambda["mu"];
  double delta1_t = param_tau["delta1_t"];
  double delta2_t = param_tau["delta2_t"];

  delta1_t = (N + P) / 2.0 + a_tau - 1.0;
  delta2_t = -b_tau - 1.0 / 2.0 * (
    N * 1.0 / S * (
      (y_n - Eigen::VectorXd::Ones(S) * mu_b0 - X_n * mu_b).squaredNorm() +
      S * sigma2_b0 +
      (X_n.transpose() * X_n * msigma_b).trace()
    ) +
    mu_lambda * (
      mu_gamma.array() * (mu_b.array().square() + msigma_b.diagonal().array())
    ).sum()
  );

  param_tau["delta1_t"] = delta1_t;
  param_tau["delta2_t"] = delta2_t;
  return(param_tau);
}

