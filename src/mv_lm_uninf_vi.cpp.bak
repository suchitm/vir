#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include "typedefs.h"
#include "include/helpers.hpp"
#include "include/multivariate_helpers.hpp"

using namespace Eigen;
using namespace Rcpp;
using namespace std;

// **********************************************************************
// individual updater functions
// **********************************************************************
Rcpp::List mvlm_vi_psi(
  Eigen::MatrixXd& E_hat, Rcpp::List& param_theta, Rcpp::List& param_tau,
  int& S, int& M, int& K, Rcpp::List& param_psi
){
  // things to use throughout
  Eigen::VectorXd mu_tau = param_tau["mu"];
  Eigen::MatrixXd mu_theta = param_theta["mu"];
  Eigen::MatrixXd msigma_theta = param_theta["msigma_mat"];
  Eigen::MatrixXd delta1_t = param_psi["delta1_t"];
  Eigen::MatrixXd delta2_t = param_psi["delta2_t"];

  // summing covariances of estimated thetas
  Eigen::MatrixXd sum_tau_msigma_theta = Eigen::MatrixXd::Constant(K, K, 0.0);
  for(int m = 0; m < M; m++)
  {
    sum_tau_msigma_theta = sum_tau_msigma_theta + mu_tau(m) *
      msigma_theta.block(m * K, 0, K, K);
  }

  // values that get re-used for each n
  Eigen::MatrixXd Dhat_tau = mu_tau.asDiagonal();
  Eigen::MatrixXd thetat_dhat_theta = mu_theta.transpose() * Dhat_tau * mu_theta;
  Eigen::MatrixXd thetat_dhat = mu_theta.transpose() * Dhat_tau;

  // only S (batch_size) psi's to update
  for(int n = 0; n < S; n++)
  {
    delta1_t.row(n) = thetat_dhat * E_hat.row(n).transpose();
    delta2_t.block(n * K, 0, K, K) = -1.0/2.0 * (
      Eigen::MatrixXd::Identity(K, K) + thetat_dhat_theta + sum_tau_msigma_theta
    );
  }
  param_psi["delta1_t"] = delta1_t;
  param_psi["delta2_t"] = delta2_t;
  return(param_psi);
}

Rcpp::List mvlm_vi_theta(
  Eigen::MatrixXd& E_hat, Rcpp::List& param_psi, Rcpp::List& param_tau,
  Rcpp::List& param_gamma, Eigen::VectorXd& mu_lambda, int& N, int& M,
  int& S, int& K, Rcpp::List& param_theta
){
  // things to use throughout
  Eigen::MatrixXd mu_psi = param_psi["mu"];
  Eigen::MatrixXd msigma_psi = param_psi["msigma_mat"];
  Eigen::MatrixXd psit_psi = mu_psi.transpose() * mu_psi;

  Eigen::MatrixXd mu_gamma = param_gamma["mu"];
  Eigen::VectorXd mu_tau = param_tau["mu"];
  Eigen::MatrixXd delta1_t = param_theta["delta1_t"];
  Eigen::MatrixXd delta2_t = param_theta["delta2_t"];

  // summing covariances of estimated psis
  Eigen::MatrixXd sum_msigma_psi = Eigen::MatrixXd::Constant(K, K, 0.0);
  for(int n = 0; n < S; n++)
  {
    sum_msigma_psi = sum_msigma_psi + msigma_psi.block(n * K, 0, K, K);
  }

  Eigen::VectorXd temp_vec = Eigen::VectorXd::Constant(K, 1.0);
  Eigen::MatrixXd Dhat_m = Eigen::MatrixXd::Constant(K, K, 1.0);

  for(int m = 0; m < M; m++)
  {
    temp_vec = mu_lambda.array() * mu_gamma.row(m).transpose().array();
    Dhat_m = temp_vec.asDiagonal();
    delta1_t.row(m) = (N * mu_tau(m)) / S * mu_psi.transpose() *
      E_hat.col(m);
    delta2_t.block(m * K, 0, K, K) = -1.0 / 2.0 / S * (
      N * mu_tau(m) * (psit_psi + sum_msigma_psi) + S * Dhat_m
    );
  }
  param_theta["delta1_t"] = delta1_t;
  param_theta["delta2_t"] = delta2_t;
  return(param_theta);
}

// b0
Rcpp::List mvlm_vi_b0(
  Eigen::MatrixXd& E_hat, Rcpp::List& param_tau, int& N, int& M, int& S,
  Rcpp::List& param_b0
){
  // values to use throughout
  Eigen::VectorXd mu_tau = param_tau["mu"];
  Eigen::VectorXd b0_delta1_t = param_b0["delta1_t"];
  Eigen::VectorXd b0_delta2_t = param_b0["delta2_t"];

  // update params
  for(int m = 0; m < M; m++)
  {
    b0_delta1_t(m) = (N * mu_tau(m)) / S * E_hat.col(m).sum();
    b0_delta2_t(m) = -1.0/2.0 * (N * mu_tau(m) + 0.000001);
  }
  param_b0["delta1_t"] = b0_delta1_t;
  param_b0["delta2_t"] = b0_delta2_t;
  return(param_b0);
}

// b
Rcpp::List mvlm_vi_b(
  Eigen::MatrixXd& E_hat, Eigen::MatrixXd& X, Rcpp::List& param_tau,
  int& N, int& M, int& P, int& S, Rcpp::List& param_b
){
  // values to use throughout
  Eigen::VectorXd mu_tau = param_tau["mu"];
  Eigen::MatrixXd b_delta1_t = param_b["delta1_t"];
  Eigen::MatrixXd b_delta2_t = param_b["delta2_t"];

  // update params
  for(int m = 0; m < M; m++)
  {
    b_delta1_t.row(m) = (N * mu_tau(m)) / S * X.transpose() * E_hat.col(m);
    b_delta2_t.block(m * P, 0, P, P) = -1.0 / 2.0 / S * (
      N * mu_tau(m) * X.transpose() * X +
      S * 0.000001 * Eigen::MatrixXd::Identity(P, P)
    );
  }
  param_b["delta1_t"] = b_delta1_t;
  param_b["delta2_t"] = b_delta2_t;
  return(param_b);
}

// tau
Rcpp::List mvlm_vi_tau (
  Eigen::MatrixXd& E_hat, Eigen::MatrixXd& X, Rcpp::List& param_b0,
  Rcpp::List& param_b, Rcpp::List& param_psi, Rcpp::List& param_theta,
  int& N, int& M, int& P, int& S, int& K, double& a_tau, double& b_tau,
  Rcpp::List& param_tau
){
  // values to use throughout
  Eigen::VectorXd mu_b0 = param_b0["mu"];
  Eigen::VectorXd vsigma2_b0 = param_b0["vsigma2"];
  Eigen::MatrixXd mu_b = param_b["mu"];
  Eigen::MatrixXd msigma_b = param_b["msigma_mat"];
  Eigen::MatrixXd mu_psi = param_psi["mu"];
  Eigen::MatrixXd msigma_psi = param_psi["msigma_mat"];
  Eigen::MatrixXd mu_theta = param_theta["mu"];
  Eigen::MatrixXd msigma_theta = param_theta["msigma_mat"];
  Eigen::MatrixXd XtX = X.transpose() * X;
  Eigen::MatrixXd psit_psi = mu_psi.transpose() * mu_psi;

  // param_vals
  Eigen::VectorXd delta1_t = param_tau["delta1_t"];
  Eigen::VectorXd delta2_t = param_tau["delta2_t"];

  // summing covariances of estimated psis
  Eigen::MatrixXd sum_msigma_psi = Eigen::MatrixXd::Constant(K, K, 0.0);
  for(int n = 0; n < S; n++)
  {
    sum_msigma_psi = sum_msigma_psi + msigma_psi.block(n * K, 0, K, K);
  }

  // update params
  for(int m = 0; m < M; m++)
  {
    double bstar_m =
      E_hat.col(m).array().square().sum() +
      S * vsigma2_b0(m) +
      (msigma_b.block(m * P, 0, P, P) * XtX).trace() + (
        msigma_theta.block(m * K, 0, K, K) * (psit_psi + sum_msigma_psi)
      ).trace() +
      mu_theta.row(m) * sum_msigma_psi * mu_theta.row(m).transpose();

    delta1_t(m) = N / 2.0 + a_tau - 1;
    delta2_t(m) = -b_tau - N / 2.0 * 1.0 / S * bstar_m;
  }

  param_tau["delta1_t"] = delta1_t;
  param_tau["delta2_t"] = delta2_t;
  return(param_tau);
}

// gamma
Rcpp::List mvlm_vi_gamma(
  Rcpp::List& param_theta, Eigen::VectorXd& mu_lambda, int& M, int& K,
  double& nu, Rcpp::List& param_gamma
){
  // values to use throughout
  Eigen::MatrixXd vsigma2_theta = param_theta["vsigma2_mat"];
  Eigen::MatrixXd mu_theta = param_theta["mu"];
  Eigen::MatrixXd ex_theta_sq = vsigma2_theta.array() +
    mu_theta.array().square();
  Eigen::MatrixXd delta1_t = param_gamma["delta1_t"];
  Eigen::MatrixXd delta2_t = param_gamma["delta2_t"];

  // update parameters
  for(int m = 0; m < M; m++)
  {
    for(int k = 0; k < K; k++)
    {
      delta1_t(m, k) = (nu + 1.0) / 2.0 - 1.0;
      delta2_t(m, k) = -1.0/2.0 * (
        nu + mu_lambda(k) * ex_theta_sq(m, k)
      );
    }
  }
  param_gamma["delta1_t"] = delta1_t;
  param_gamma["delta2_t"] = delta2_t;
  return(param_gamma);
}

// xi
Rcpp::List mvlm_vi_xi(
  Rcpp::List& param_gamma, Rcpp::List& param_theta, int& M, int& K,
  Eigen::VectorXd& mu_lambda, double& a1, double& a2, Rcpp::List& param_xi,
  bool svb = false
){
  // values to use throughout
  Eigen::MatrixXd mu_gamma = param_gamma["mu"];
  Eigen::MatrixXd mu_theta = param_theta["mu"];
  Eigen::MatrixXd vsigma2_theta = param_theta["vsigma2_mat"];
  Eigen::VectorXd mu_xi = param_xi["mu"];
  Eigen::VectorXd delta1_t = param_xi["delta1_t"];
  Eigen::VectorXd delta2_t = param_xi["delta2_t"];

  // store expectation product
  Eigen::VectorXd prod_vec = (
    mu_gamma.array() * (mu_theta.array().square() + vsigma2_theta.array())
  ).colwise().sum();

  // xi1
  delta1_t(0) = a1 - 1.0 + (M * K) / 2.0;
  delta2_t(0) = -1.0 - 1.0 / 2.0 / mu_xi(0) *
    (mu_lambda.array() * prod_vec.array()).sum();
  if(svb == false)
  {
    param_xi["delta1_t"] = delta1_t;
    param_xi["delta2_t"] = delta2_t;
    vi_update_gamma(param_xi, 1.0);
    natural_to_canonical(param_xi, "gamma");
    mu_xi = param_xi["mu"];
    mu_lambda = cum_prod(mu_xi);
  }

  // update xi(2:K)
  for(int k = 1; k < K; k++)
  {
    delta1_t(k) = a2 - 1.0 + M / 2.0 * (K - k);
    delta2_t(k) = -1.0 - 1.0 / 2.0 / mu_xi(k) *
      (mu_lambda.tail(K - k).array() * prod_vec.tail(K - k).array()).sum();
    if(svb == false)
    {
      param_xi["delta1_t"] = delta1_t;
      param_xi["delta2_t"] = delta2_t;
      vi_update_gamma(param_xi, 1.0);
      natural_to_canonical(param_xi, "gamma");
      mu_xi = param_xi["mu"];
      mu_lambda = cum_prod(mu_xi);
    }
  }
  param_xi["delta1_t"] = delta1_t;
  param_xi["delta2_t"] = delta2_t;
  return(param_xi);
}

double mv_lm_uninf_elbo(
  Eigen::MatrixXd& X_s, Eigen::MatrixXd& Y_s,
  Rcpp::List& param_psi, Rcpp::List& param_theta, Rcpp::List& param_b0,
  Rcpp::List& param_b, Rcpp::List& param_tau, Rcpp::List& param_gamma,
  Rcpp::List& param_xi, double& a_tau, double& b_tau, int& N, int& S, int& M,
  int& P, int& K
){
  Eigen::VectorXd one_S = Eigen::VectorXd::Constant(S, 1.0);
  // ************************************************************************
  // load the variable for use in functions
  // ************************************************************************
  Eigen::VectorXd mu_psi = param_psi["mu"];
  Eigen::VectorXd msigma_mat_psi = param_psi["msigma_mat"];
  Eigen::MatrixXd vsigma2_mat_psi = param_psi["vsigma2_mat"];
  Eigen::VectorXd logdet_msigma_psi = param_psi["logdet_msigma"];

  Eigen::VectorXd mu_theta = param_theta["mu"];
  Eigen::VectorXd msigma_mat_theta = param_theta["msigma_mat"];
  Eigen::MatrixXd vsigma2_mat_theta = param_theta["vsigma2_mat"];
  Eigen::VectorXd logdet_msigma_theta = param_theta["logdet_msigma"];

  Eigen::VectorXd mu_b0 = param_b0["mu"];
  Eigen::VectorXd msigma_b0 = param_b0["msigma"];
  Eigen::VectorXd vsigma2_b0 = param_b0["logdet_msigma"];

  Eigen::VectorXd mu_b = param_b["mu"];
  Eigen::VectorXd msigma_mat_b = param_b["msigma_mat"];
  Eigen::MatrixXd vsigma2_mat_b = param_b["vsigma2_mat"];
  Eigen::VectorXd logdet_msigma_b = param_b["logdet_msigma"];

  Eigen::VectorXd astar_tau = param_tau["shape"];
  Eigen::VectorXd bstar_tau = param_tau["rate"];
  Eigen::VectorXd mu_tau = param_tau["mu"];
  Eigen::VectorXd mu_log_tau = param_tau["mu_log"];

  // ************************************************************************
  // calculate elbo
  // ************************************************************************
  // ----- log lik -----
  Eigen::MatrixXd E_hat = Y_s - one_S * mu_b0.transpose() -
    X_s * mu_b.transpose() - mu_psi * mu_theta.transpose();
  Eigen::MatrixXd D_tau = mu_tau.asDiagonal();

  //  add log 2 pi and then the taus
  double ll = -N/2.0 * M * std::log(2.0 * M_PI);
  for(int m = 0; m < M; m++)
    ll = ll + N/2.0 * (Rf_digamma(astar_tau(m)) - std::log(bstar_tau(m)));
  // matrix normal kernel
  ll = ll - 1.0/2.0 * (D_tau * E_hat.transpose() * E_hat).trace();

  // ----- psi -----
  double lp_psi = 0.0;
  double lq_psi = 0.0;
  Eigen::MatrixXd mu_prec = Eigen::MatrixXd::Identity(K, K);
  double mu_logdet_prec = 0.0;
  Eigen::MatrixXd this_msigma_psi = Eigen::MatrixXd::Identity(K, K);
  Eigen::VectorXd this_mu_psi;

  for(int s = 0; s < S; s++)
  {
    this_mu_psi = mu_psi.row(s).transpose();
    this_msigma_psi = msigma_mat_psi.block(s * K, 0, K, K);
    mu_prec = mu_xi

    // lp
    lp_psi +=
      -K / 2.0 * std::log(2.0 * M_PI) +
      1.0 / 2.0 * mu_logdet_prec -
      1.0 / 2.0 * (
        this_mu_psi.transpose() * mu_prec * this_mu_psi +
        (mu_prec * this_msigma_psi).trace()
      );

    // lq
    lq_psi += -1.0/2.0 * logdet_msigma_psi(s) -
      K/2.0 * (1 + std::log(2.0 * M_PI));
  }

  // ----- theta -----
  double lp_theta = 0.0;
  double lq_theta = 0.0;
  mu_prec = Eigen::MatrixXd::Identity(K, K);
  mu_logdet_prec = 0.0;
  Eigen::MatrixXd this_msigma_theta = Eigen::MatrixXd::Identity(K, K);
  Eigen::VectorXd this_mu_theta;

  for(int m = 0; m < M; m++)
  {
    this_mu_theta = mu_theta.row(m).transpose();
    this_msigma_theta = msigma_mat_theta.block(m * K, 0, K, K);

    // lp
    lp_theta +=
      -K / 2.0 * std::log(2.0 * M_PI) +
      1.0 / 2.0 * mu_logdet_prec -
      1.0 / 2.0 * (
        this_mu_theta.transpose() * mu_prec * this_mu_theta +
        (mu_prec * this_msigma_theta).trace()
      );

    // lq
    lq_theta += -1.0/2.0 * logdet_msigma_theta(m) -
      K/2.0 * (1 + std::log(2.0 * M_PI));
  }

  // ----- b0 -----
  Eigen::MatrixXd mu_prec_b0 = Eigen::MatrixXd::Identity(M, M);
  double mu_logdet_prec_b0 = 0.0;
  double lp_b0 = lp_mv_normal(mu_prec, mu_logdet_prec, param_b0, M);
  double lq_b0 = lq_mv_normal(param_b, M);

  // ----- b -----
  double lp_b = 0.0;
  double lq_b = 0.0;
  mu_prec = 0.000001 * Eigen::MatrixXd::Identity(K, K);
  mu_logdet_prec = 0.0;
  Eigen::MatrixXd this_msigma_b = Eigen::MatrixXd::Identity(K, K);
  Eigen::VectorXd this_mu_b;

  for(int m = 0; m < M; m++)
  {
    this_mu_b = mu_b.row(m).transpose();
    this_msigma_b = msigma_mat_b.block(m * P, 0, P, P);

    // lp
    lp_b +=
      -P/2.0 * std::log(2.0 * M_PI) +
      1.0 / 2.0 * mu_logdet_prec -
      1.0 / 2.0 * (
        this_mu_b.transpose() * mu_prec * this_mu_b +
        (mu_prec * this_msigma_b).trace()
      );

    // lq
    lq_b += -1.0/2.0 * logdet_msigma_b(m) -
      P/2.0 * (1 + std::log(2.0 * M_PI));
  }

  // ----- tau -----
  double lp_tau = 0.0;
  double lq_tau = 0.0;
  for(int m = 0; m < M; m++)
  {
    // lp
    lp_tau +=
      a_tau * std::log(b_tau) - Rf_lgammafn(a_tau) +
      (a_tau - 1) * (Rf_digamma(astar_tau(m)) - std::log(bstar_tau(m))) -
      b_tau * mu_tau(m);

    // lq
    lq_tau +=
      -Rf_lgammafn(astar_tau(m)) + std::log(bstar_tau(m)) +
      (astar_tau(m) - 1.0) * Rf_digamma(astar_tau(m)) - astar_tau(m);
  }

  // ----- gamma -----
  // ----- xi -----


  return(1.0);


}

// **********************************************************************
// CAVI
// **********************************************************************
// [[Rcpp::export]]
Rcpp::List mv_lm_uninf_cavi_cpp(
  Eigen::MatrixXd Y, Eigen::MatrixXd X, int K, int n_iter,
  double rel_tol = 0.0001, bool verbose = true, double a_tau = 0.1,
  double b_tau = 0.1
){
  // problem info
  int N = Y.rows();
  int M = Y.cols();
  int P = X.cols();
  int S = N;
  Eigen::VectorXd one_N = Eigen::VectorXd::Constant(N, 1.0);

  // initialize parameters
  Rcpp::List param_b0 = vi_init_mv_normal(M, 1);
  Rcpp::List param_b = vi_init_indep_matrix_normal(M, P, 0.0);
  Rcpp::List param_theta = vi_init_indep_matrix_normal(M, K, 0.1);
  Rcpp::List param_psi = vi_init_indep_matrix_normal(N, K, 0.1);
  Rcpp::List param_gamma = vi_init_indep_matrix_gamma(M, K);
  Rcpp::List param_tau = vi_init_gamma(M);
  Rcpp::List param_xi = vi_init_gamma(K);

  // start with hard shrinkage on the elements of mtheta
  Eigen::VectorXd mu_xi = param_xi["mu"];
  for(int k = 0; k < K; k++)
  {
    mu_xi(k) = 1.0 + k / 2.0;
  }
  param_xi["shape"] = mu_xi;
  param_xi["mu"] = mu_xi;
  Eigen::VectorXd mu_lambda = cum_prod(mu_xi);

  // other hyperparams
  double a1 = 2.1; double a2 = 10.1;
  double nu = 3.0;
  double rhot = 1.0;
  Eigen::MatrixXd E_hat(N, M);
  Eigen::MatrixXd mu_b(M, P);
  Eigen::VectorXd mu_b0(M);
  Eigen::MatrixXd mu_psi(S, K);
  Eigen::MatrixXd mu_theta(M, K);

  for(int i = 0; i < n_iter; i++)
  {
    // check interrupt and print progress
    Rcpp::checkUserInterrupt();
    if(verbose && (i % 100 == 0)) {
        Rcout << "Done with Iteration " << i << " of " << n_iter << "\r";
    }

    // psi
    mu_b0 = param_b0["mu"];
    mu_b = param_b["mu"];
    E_hat = Y - one_N * mu_b0.transpose() - X * mu_b.transpose();
    mvlm_vi_psi(E_hat, param_theta, param_tau, S, M, K, param_psi);
    vi_update_indep_matrix_normal(param_psi, rhot);
    natural_to_canonical(param_psi, "matrix_normal_ind");
    mu_psi = param_psi["mu"];

    // theta
    mvlm_vi_theta(
      E_hat, param_psi, param_tau, param_gamma, mu_lambda, N, M, S, K,
      param_theta
    );
    vi_update_indep_matrix_normal(param_theta, rhot);
    natural_to_canonical(param_theta, "matrix_normal_ind");
    mu_theta = param_theta["mu"];


    // b0
    E_hat = Y - X * mu_b.transpose() - mu_psi * mu_theta.transpose();
    mvlm_vi_b0(E_hat, param_tau, N, M, S, param_b0);
    vi_update_mv_normal(param_b0, 1, rhot);
    natural_to_canonical(param_b0, "mv_normal_ind");
    mu_b0 = param_b0["mu"];

    // b
    E_hat = Y - one_N * mu_b0.transpose() - mu_psi * mu_theta.transpose();
    mvlm_vi_b(E_hat, X, param_tau, N, M, P, S, param_b);
    vi_update_indep_matrix_normal(param_b, rhot);
    natural_to_canonical(param_b, "matrix_normal_ind");
    mu_b = param_b["mu"];

    // tau
    E_hat = Y - one_N * mu_b0.transpose() - X * mu_b.transpose() -
      mu_psi * mu_theta.transpose();
    mvlm_vi_tau(
      E_hat, X, param_b0, param_b, param_psi, param_theta, N, M, P, S, K, a_tau,
      b_tau, param_tau
    );
    vi_update_gamma(param_tau, rhot);
    natural_to_canonical(param_tau, "gamma");

    // xi
    mvlm_vi_xi(
      param_gamma, param_theta, M, K, mu_lambda, a1, a2, param_xi, false
    );

    // gamma
    mvlm_vi_gamma(param_theta, mu_lambda, M, K, nu, param_gamma);
    vi_update_indep_matrix_gamma(param_gamma, rhot);
    natural_to_canonical(param_gamma, "matrix_gamma_ind");

  }

  Eigen::VectorXd vsigma2_b0 = param_b0["vsigma2"];
  Eigen::MatrixXd msigma_b = param_b["msigma_mat"];
  Eigen::MatrixXd msigma_theta = param_theta["msigma_mat"];

  Rcpp::List retl;
  retl["vmu_b0"] = mu_b0;
  retl["vsigma2_b0"] = vsigma2_b0;
  retl["mu_mat_b"] = mu_b;
  retl["msigma_mat_b"] = msigma_b;
  retl["mu_mat_theta"] = param_theta["mu"];
  retl["msigma_mat_theta"] = msigma_theta;
  retl["param_tau"] = param_tau;

  return(retl);
}

// **********************************************************************
// SVI
// **********************************************************************
// [[Rcpp::export]]
Rcpp::List mvlm_uninf_svi_cpp(
  Eigen::MatrixXd Y, Eigen::MatrixXd X, int K, int n_iter, bool verbose = true,
  double a_tau = 0.1, double b_tau = 0.1, int batch_size = 42,
  double const_rhot = 0.01, double omega = 15.0, double kappa = 0.6
){
  // problem info
  int N = Y.rows();
  int M = Y.cols();
  int P = X.cols();
  int S = batch_size;
  Eigen::VectorXd one_S = Eigen::VectorXd::Constant(S, 1.0);

  // initialize parameters
  Rcpp::List param_b0 = vi_init_mv_normal(M, 1);
  Rcpp::List param_b = vi_init_indep_matrix_normal(M, P, 0.0);
  Rcpp::List param_theta = vi_init_indep_matrix_normal(M, K, 0.1);
  Rcpp::List param_psi = vi_init_indep_matrix_normal(S, K, 0.1);
  Rcpp::List param_gamma = vi_init_indep_matrix_gamma(M, K);
  Rcpp::List param_tau = vi_init_gamma(M);
  Rcpp::List param_xi = vi_init_gamma(K);
  Eigen::VectorXd mu_lambda = Eigen::VectorXd::Constant(K, 1.0);

  // temp storage
  Eigen::MatrixXd E_hat(S, M);
  Eigen::VectorXd mu_b0(M);
  Eigen::MatrixXd mu_b(M, P);
  Eigen::MatrixXd mu_psi(S, K);
  Eigen::MatrixXd mu_theta(M, K);
  Eigen::MatrixXd Y_s(S, M);
  Eigen::MatrixXd X_s(S, P);
  Eigen::VectorXd mu_xi(K);

  Rcpp::IntegerVector the_sample = seq(0, S - 1);
  Rcpp::IntegerVector seq_samp = seq(0, N - 1);
  double rhot = 0.0;

  // other hyperparams
  double a1 = 2.1; double a2 = 3.1;
  double nu = 0.5;

  for(int i = 0; i < n_iter; i++)
  {

    if(const_rhot <= 0)
      rhot = std::exp(-kappa * std::log(i + 1.0 + omega));
    else
      rhot = const_rhot;

    Rcpp::checkUserInterrupt();
    if(verbose && (i % 10 == 0)) {
      Rcpp::Rcout << "Done with Iteration " << i << " of " << n_iter <<
        " with step size " << rhot << "\r";
    }

    // means for error calculations
    mu_b0 = param_b0["mu"];
    mu_b = param_b["mu"];
    mu_theta = param_theta["mu"];
    mu_xi = param_xi["mu"];

    // sample data
    the_sample = sample(seq_samp, S, false);
    get_subsample_mvlm(Y, X, Y_s, X_s, S, N, the_sample);

    // psi
    E_hat = Y_s - one_S * mu_b0.transpose() - X_s * mu_b.transpose();
    mvlm_vi_psi(E_hat, param_theta, param_tau, S, M, K, param_psi);
    vi_update_indep_matrix_normal(param_psi, 1.0);
    natural_to_canonical(param_psi, "matrix_normal_ind");
    mu_psi = param_psi["mu"];

    // theta
    mvlm_vi_theta(
      E_hat, param_psi, param_tau, param_gamma, mu_lambda, N, M, S, K,
      param_theta
    );

    // b0
    E_hat = Y_s - X_s * mu_b.transpose() - mu_psi * mu_theta.transpose();
    mvlm_vi_b0(E_hat, param_tau, N, M, S, param_b0);

    // b
    E_hat = Y_s - one_S * mu_b0.transpose() - mu_psi * mu_theta.transpose();
    mvlm_vi_b(E_hat, X_s, param_tau, N, M, P, S, param_b);

    // tau
    E_hat = Y_s - one_S * mu_b0.transpose() - X_s * mu_b.transpose() -
      mu_psi * mu_theta.transpose();
    mvlm_vi_tau(
      E_hat, X_s, param_b0, param_b, param_psi, param_theta, N, M, P, S, K,
      a_tau, b_tau, param_tau
    );

    // gamma
    mvlm_vi_gamma(param_theta, mu_lambda, M, K, nu, param_gamma);

    // xi
    mvlm_vi_xi(
      param_gamma, param_theta, M, K, mu_lambda, a1, a2, param_xi, true
    );

    // update and convert parametrs
    vi_update_indep_matrix_normal(param_theta, rhot);
    vi_update_mv_normal(param_b0, 1, rhot);
    vi_update_indep_matrix_normal(param_b, rhot);
    vi_update_gamma(param_tau, rhot);
    vi_update_indep_matrix_gamma(param_gamma, rhot);
    vi_update_gamma(param_xi, rhot);

    // get means and variances of parameters
    natural_to_canonical(param_theta, "matrix_normal_ind");
    natural_to_canonical(param_b0, "mv_normal_ind");
    natural_to_canonical(param_b, "matrix_normal_ind");
    natural_to_canonical(param_tau, "gamma");
    natural_to_canonical(param_gamma, "matrix_gamma_ind");
    natural_to_canonical(param_xi, "gamma");

    // update mu_lambda
    mu_xi = param_xi["mu"];
    mu_lambda = cum_prod(mu_xi);
  }

  Eigen::VectorXd vsigma2_b0 = param_b0["vsigma2"];
  Eigen::MatrixXd msigma_b = param_b["msigma_mat"];
  Eigen::MatrixXd msigma_theta = param_theta["msigma_mat"];

  Rcpp::List retl;
  retl["vmu_b0"] = mu_b0;
  retl["vsigma2_b0"] = vsigma2_b0;
  retl["mu_mat_b"] = mu_b;
  retl["msigma_mat_b"] = msigma_b;
  retl["mu_mat_theta"] = param_theta["mu"];
  retl["msigma_mat_theta"] = msigma_theta;
  retl["param_tau"] = param_tau;

  return(retl);
}


