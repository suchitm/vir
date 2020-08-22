#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include "typedefs.h"
#include "include/helpers.hpp"

// *****************************************************************************
// Variational Algorithm Helpers
// *****************************************************************************
Rcpp::List lm_vi_b0(
  Eigen::VectorXd& y_s, Eigen::MatrixXd& X_s, Rcpp::List& param_b,
  Rcpp::List& param_tau, int& N, int& S, Rcpp::List& param_b0
){
  double mu_tau = param_tau["mu"];
  Eigen::VectorXd mu_b = param_b["mu"];

  double delta1_t = N * 1.0 / S * mu_tau * (y_s - X_s * mu_b).array().sum();
  double delta2_t = - 1.0 / 2.0 * (N * mu_tau + .000001);

  param_b0["delta1_t"] = delta1_t;
  param_b0["delta2_t"] = delta2_t;

  return(param_b0);
}

Rcpp::List lm_vi_b(
  Eigen::VectorXd& y_s, Eigen::MatrixXd& X_s, Rcpp::List& param_b0,
  Rcpp::List& param_tau, Eigen::MatrixXd& mu_prior_mat, int& N, int& S,
  int& P, int& type, bool cavi, Rcpp::List& param_b
){
  Eigen::VectorXd delta1_t = param_b["delta1_t"];
  Eigen::MatrixXd delta2_t0(P, P);
  Eigen::VectorXd delta2_t1(P);
  double mu_tau = param_tau["mu"];
  double mu_b0 = param_b0["mu"];
  Eigen::VectorXd one_S = Eigen::VectorXd::Constant(S, 1.0);
  Eigen::VectorXd ehat_np(S);
  Eigen::VectorXd mu_b = param_b["mu"];

  if(type == 0)
  {
    delta1_t = N * 1.0 * mu_tau / S * X_s.transpose() * (y_s - one_S * mu_b0);
    delta2_t0 = -mu_tau / 2.0 / S *
      (N * X_s.transpose() * X_s + S * mu_prior_mat);
    param_b["delta2_t"] = delta2_t0;
  }
  else
  {
    for(int p = 0; p < P; p++)
    {
      ehat_np = y_s - one_S * mu_b0 - X_s.leftCols(p) * mu_b.head(p) -
        X_s.rightCols(P - p - 1) * mu_b.tail(P - p - 1);
      delta1_t(p) = N * mu_tau / S * X_s.col(p).transpose() * ehat_np;
      delta2_t1(p) = -mu_tau / 2.0 / S * (
        N * X_s.col(p).array().square().sum() + S * mu_prior_mat(p, p)
      );
      if(cavi) {
        mu_b(p) = -1.0 / 2.0 * delta1_t(p) / delta2_t1(p);
      }
    }
    param_b["delta2_t"] = delta2_t1;
  }
  param_b["delta1_t"] = delta1_t;
  return(param_b);
}
