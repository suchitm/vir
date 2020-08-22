#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include "typedefs.h"
#include "include/tuvn_helpers.hpp"
#include "include/helpers.hpp"

// *****************************************************************************
// Gibbs sampler helpers
// *****************************************************************************
// [[Rcpp::export]]
Eigen::VectorXd probit_gibbs_z(
  Eigen::VectorXi& y, Eigen::VectorXd& eta, int& N, Eigen::VectorXd& z
){
  for(int i = 0; i < N; i++)
  {
    if(y(i) == 1)
      z(i) = rtuvn(1, eta(i), 1.0, 0.0, R_PosInf)(0);
    else
      z(i) = rtuvn(1, eta(i), 1.0, R_NegInf, 0.0)(0);
  }
  return(z);
}

// [[Rcpp::export]]
double probit_gibbs_b0(Eigen::VectorXd& ehat, int& N, double& b0)
{
  double G = (N + 0.00001);
  double g = ehat.array().sum();
  double mu = g / G;
  double sd = std::sqrt(1.0 / G);
  b0 = Rcpp::rnorm(1, mu, sd)(0);
  return(b0);
}

// [[Rcpp::export]]
Eigen::VectorXd probit_gibbs_b(
  Eigen::MatrixXd& X, Eigen::VectorXd& ehat, Eigen::MatrixXd& prior_mat, 
  int& P, Eigen::VectorXd& b
){
  Eigen::MatrixXd G = X.transpose() * X + prior_mat;
  Eigen::VectorXd g = X.transpose() * ehat;
  Eigen::LLT<Eigen::MatrixXd> chol_G(G);
  Eigen::VectorXd mu(chol_G.solve(g));
  b = mu + chol_G.matrixU().solve(conv(Rcpp::rnorm(P, 0, 1)));
  return(b);
}

// [[Rcpp::export]]
double probit_log_lik(
  Eigen::VectorXi& y, Eigen::MatrixXd& X, double& b0, Eigen::VectorXd& b, 
  int& N
){
  Eigen::VectorXd eta = b0 * Eigen::VectorXd::Constant(N, 1) + X * b;
  double log_lik = 0.0; 
  double p_y1;
  for(int n = 0; n < N; n++)
  {
    p_y1 = R::pnorm(eta(n), 0.0, 1.0, true, false);
    log_lik += y(n) * std::log(p_y1) + (1 - y(n)) * std::log(1.0 - p_y1);
  }
  return(log_lik);
}

// *****************************************************************************
// Variational Algorithm Helpers
// *****************************************************************************
// [[Rcpp::export]]
Rcpp::List probit_vi_z(
  Eigen::MatrixXd& X_s, Rcpp::List& param_b0, Rcpp::List& param_b,
  int& S, Rcpp::List& param_z
){

  double mu_b0 = param_b0["mu"];
  Eigen::VectorXd mu_b = param_b["mu"];
  Eigen::VectorXd ones = Eigen::VectorXd::Constant(S, 1.0);

  Eigen::VectorXd delta1_t = mu_b0 * ones + X_s * mu_b;
  Eigen::VectorXd delta2_t = -1.0 / 2.0 * ones;

  param_z["delta1_t"] = delta1_t;
  param_z["delta2_t"] = delta2_t;

  return(param_z);
}

// [[Rcpp::export]]
Rcpp::List probit_vi_b0(
  Eigen::MatrixXd& X_s, Rcpp::List& param_z, Rcpp::List& param_b, int& N,
  int& S, Rcpp::List& param_b0
){
  Eigen::VectorXd mu_z = param_z["mu"];
  Eigen::VectorXd mu_b = param_b["mu"];
  Eigen::VectorXd ones = Eigen::VectorXd::Constant(S, 1.0);

  double delta1_t = N * 1.0 / S * (mu_z - X_s * mu_b).array().sum();
  double delta2_t = - (N + .000001) / 2.0;

  param_b0["delta1_t"] = delta1_t;
  param_b0["delta2_t"] = delta2_t;

  return(param_b0);
}

// [[Rcpp::export]]
Rcpp::List probit_vi_b(
  Eigen::MatrixXd& X_s, Rcpp::List& param_z, Rcpp::List& param_b0,
  Eigen::MatrixXd& mu_prior_mat, int& N, int& S, int& P, int& type,
  bool cavi, Rcpp::List& param_b
){
  Eigen::VectorXd delta1_t = param_b["delta1_t"];
  Eigen::MatrixXd delta2_t0(P, P);
  Eigen::VectorXd delta2_t1(P);
  Eigen::VectorXd mu_z = param_z["mu"];
  double mu_b0 = param_b0["mu"];
  Eigen::VectorXd one_S = Eigen::VectorXd::Constant(S, 1.0);
  Eigen::VectorXd ehat_np(S);
  Eigen::VectorXd mu_b = param_b["mu"];

  if(type == 0)
  {
    delta1_t = N * 1.0 / S * X_s.transpose() * (mu_z - one_S * mu_b0);
    delta2_t0 = -1.0 / 2.0 / S *
      (N * X_s.transpose() * X_s + S * mu_prior_mat);
    param_b["delta2_t"] = delta2_t0;
  }
  else
  {
    for(int p = 0; p < P; p++)
    {
      ehat_np = mu_z - one_S * mu_b0 - X_s.leftCols(p) * mu_b.head(p) -
        X_s.rightCols(P - p - 1) * mu_b.tail(P - p - 1);
      delta1_t(p) = N * 1.0 / S * X_s.col(p).transpose() * ehat_np;
      delta2_t1(p) = -1.0 / 2.0 / S * (
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

