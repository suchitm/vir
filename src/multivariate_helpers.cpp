#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include "typedefs.h"
#include "include/tuvn_helpers.hpp"
#include "include/probit_helpers.hpp"
#include "include/helpers.hpp"

//**********************************************************************//
// other helpers
//**********************************************************************//
Eigen::VectorXd cum_prod(Eigen::VectorXd& x)
{
  int n = x.size();
  Eigen::VectorXd out_vec = Eigen::VectorXd::Constant(n, 1.0);
  double prod = 1.0;
  for(int i = 0; i < n; i++)
  {
    prod = prod * x(i);
    out_vec(i) = prod;
  }
  return(out_vec);
}

//**********************************************************************//
// individual samplers
//**********************************************************************//
// [[Rcpp::export]]
Eigen::MatrixXd mv_probit_gibbs_Z(
  Eigen::MatrixXi& Y, Eigen::MatrixXd& Eta, Eigen::VectorXd& tau, int& N,
  int& M, Eigen::MatrixXd& Z
){
  double this_sd;
  for(int m = 0; m < M; m++)
  {
    this_sd = 1.0 / std::sqrt(tau(m));
    for(int n = 0; n < N; n++)
    {
      if(Y(n, m) == 1)
        Z(n, m) = rtuvn(1, Eta(n, m), this_sd, 0.0, R_PosInf)(0);
      else
        Z(n, m) = rtuvn(1, Eta(n, m), this_sd, R_NegInf, 0.0)(0);
    }
  }
  return(Z);
}

// [[Rcpp::export]]
Eigen::VectorXd mvlm_uninf_gibbs_b0(
  Eigen::MatrixXd& E_hat, Eigen::VectorXd& tau, int& N, int& M,
  Eigen::VectorXd& b0
){
  for(int m = 0; m < M; m++)
  {
    double g = tau(m) * E_hat.col(m).sum();
    double G = N * tau(m) + 0.000001;
    double mu = g / G;
    double sd = std::sqrt(1.0 / G);
    b0(m) = Rcpp::rnorm(1, mu, sd)(0);
  }
  return(b0);
}

// [[Rcpp::export]]
Eigen::MatrixXd mvlm_uninf_gibbs_B(
  Eigen::MatrixXd& E_hat, Eigen::MatrixXd& X, Eigen::VectorXd& tau, int& M,
  int& P, Eigen::MatrixXd& b
){
  Eigen::MatrixXd XtX = X.transpose() * X;
  for(int m = 0; m < M; m++)
  {
    Eigen::MatrixXd G = tau(m) * XtX +
      0.000001 * Eigen::MatrixXd::Identity(P, P);
    Eigen::VectorXd g = tau(m) * X.transpose() * E_hat.col(m);
    Eigen::LLT<Eigen::MatrixXd> chol_G(G);
    Eigen::VectorXd mu(chol_G.solve(g));
    b.row(m) = mu + chol_G.matrixU().solve(conv(Rcpp::rnorm(P, 0, 1)));
  }
  return(b);
}

// [[Rcpp::export]]
Eigen::MatrixXd mvlm_uninf_gibbs_mtheta(
  Eigen::MatrixXd& E_hat, Eigen::MatrixXd& mphi, Eigen::MatrixXd& mgamma,
  Eigen::VectorXd& tau, Eigen::VectorXd& lambda, int& M, int& K,
  Eigen::MatrixXd& mtheta
){
  Eigen::MatrixXd mphi_t_mphi = mphi.transpose() * mphi;
  for(int m = 0; m < M; m++)
  {
    Eigen::MatrixXd G = tau(m) * mphi_t_mphi;
    Eigen::VectorXd temp = lambda.array() * mgamma.row(m).transpose().array();
    G += temp.asDiagonal();
    Eigen::VectorXd g = tau(m) * mphi.transpose() * E_hat.col(m);
    Eigen::LLT<Eigen::MatrixXd> chol_G(G);
    Eigen::VectorXd mu(chol_G.solve(g));
    mtheta.row(m) = mu + chol_G.matrixU().solve(conv(Rcpp::rnorm(K, 0, 1)));
  }
  return(mtheta);
}

// [[Rcpp::export]]
Eigen::MatrixXd mvlm_uninf_gibbs_mphi(
  Eigen::MatrixXd& E_hat, Eigen::MatrixXd& mtheta, Eigen::VectorXd& tau,
  int& N, int& K, Eigen::MatrixXd& mphi
){
  Eigen::MatrixXd D_tau = tau.asDiagonal();
  Eigen::MatrixXd mthetat_D_mtheta = mtheta.transpose() * D_tau * mtheta;
  Eigen::MatrixXd I_K = Eigen::MatrixXd::Identity(K, K);
  for(int n = 0; n < N; n++)
  {
    Eigen::VectorXd g = mtheta.transpose() * D_tau * E_hat.row(n).transpose();
    Eigen::MatrixXd G = mthetat_D_mtheta + I_K;
    Eigen::LLT<Eigen::MatrixXd> chol_G(G);
    Eigen::VectorXd mu(chol_G.solve(g));
    mphi.row(n) = mu + chol_G.matrixU().solve(conv(Rcpp::rnorm(K, 0, 1)));
  }
  return(mphi);
}

// [[Rcpp::export]]
Eigen::VectorXd mvlm_uninf_gibbs_tau(
  Eigen::MatrixXd& E_hat, int& N, int& M, double& a_tau, double& b_tau,
  Eigen::VectorXd& tau
){
  for(int m = 0; m < M; m++)
  {
    double shape = N/2.0 + a_tau;
    double rate = b_tau + 1.0/2.0 * E_hat.col(m).array().square().sum();
    tau(m) = Rcpp::rgamma(1, shape, 1/rate)(0);
  }
  return(tau);
}

// [[Rcpp::export]]
Eigen::MatrixXd mvlm_uninf_gibbs_mgamma(
  Eigen::MatrixXd& mtheta, Eigen::VectorXd& lambda, int& M, int& K,
  double& nu, Eigen::MatrixXd& mgamma
){
  for(int m = 0; m < M; m++)
  {
    for(int k = 0; k < K; k++)
    {
      double shape = (nu + 1.0) / 2.0;
      double rate = 1.0 / 2.0 * (nu + lambda(k) * mtheta(m, k) * mtheta(m, k));
      mgamma(m, k) = Rcpp::rgamma(1, shape, 1/rate)(0);
    }
  }
  return(mgamma);
}

// [[Rcpp::export]]
Eigen::VectorXd mvlm_uninf_gibbs_xi(
  Eigen::MatrixXd& mtheta, Eigen::MatrixXd& mgamma, Eigen::VectorXd& lambda,
  int& M, int& K, double& a1, double& a2, Eigen::VectorXd& xi
){
  // update xi_1
  Eigen::VectorXd prod_vec =
    (mtheta.array().square() * mgamma.array()).colwise().sum();

  double shape = a1 + (M * K) / 2.0;
  double rate = 1 + 1 / xi(0) * 1/2 * (
    (lambda.array() * prod_vec.array()).sum()
  );
  xi(0) = Rcpp::rgamma(1, shape, 1/rate)(0);
  lambda = cum_prod(xi);

  // update xi[2:K]
  for(int k = 1; k < K; k++)
  {
    shape = a2 + M / 2.0 * (K - k);
    rate = 1 + 1.0 / 2.0 * 1.0 / xi(k) *
      (lambda.tail(K - k).array() * prod_vec.tail(K - k).array()).sum();
    xi(k) = Rcpp::rgamma(1, shape, 1/rate)(0);
    lambda = cum_prod(xi);
  }

  return(xi);
}

//**********************************************************************//
// variational algorithms
//**********************************************************************//
// [[Rcpp::export]]
Rcpp::List mv_probit_vi_z(
  Eigen::MatrixXd& Eta, Rcpp::List& param_tau, int& S, int& M,
  Rcpp::List& param_z
){
  Eigen::MatrixXd delta1_t(S, M);
  Eigen::MatrixXd delta2_t(S, M);
  Eigen::VectorXd mu_tau = param_tau["mu"];

  for(int m = 0; m < M; m++)
  {
    for(int s = 0; s < S; s++)
    {
      delta1_t(s, m) = mu_tau(m) * Eta(s, m);
      delta2_t(s, m) = -mu_tau(m) / 2.0;
    }
  }
  param_z["delta1_t"] = delta1_t;
  param_z["delta2_t"] = delta2_t;
}

// [[Rcpp::export]]
Rcpp::List mvlm_vi_phi(
  Eigen::MatrixXd& E_hat, Rcpp::List& param_theta, Rcpp::List& param_tau,
  int& S, int& M, int& K, Rcpp::List& param_phi
){
  // things to use throughout
  Eigen::VectorXd mu_tau = param_tau["mu"];
  Eigen::MatrixXd mu_theta = param_theta["mu"];
  Eigen::MatrixXd msigma_theta = param_theta["msigma_mat"];
  Eigen::MatrixXd delta1_t = param_phi["delta1_t"];
  Eigen::MatrixXd delta2_t = param_phi["delta2_t"];

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

  // only S (batch_size) phi's to update
  for(int n = 0; n < S; n++)
  {
    delta1_t.row(n) = thetat_dhat * E_hat.row(n).transpose();
    delta2_t.block(n * K, 0, K, K) = -1.0/2.0 * (
      Eigen::MatrixXd::Identity(K, K) + thetat_dhat_theta + sum_tau_msigma_theta
    );
  }
  param_phi["delta1_t"] = delta1_t;
  param_phi["delta2_t"] = delta2_t;
  return(param_phi);
}

// [[Rcpp::export]]
Rcpp::List mvlm_vi_theta(
  Eigen::MatrixXd& E_hat, Rcpp::List& param_phi, Rcpp::List& param_tau,
  Rcpp::List& param_gamma, Eigen::VectorXd& mu_lambda, int& N, int& M,
  int& S, int& K, Rcpp::List& param_theta
){
  // things to use throughout
  Eigen::MatrixXd mu_phi = param_phi["mu"];
  Eigen::MatrixXd msigma_phi = param_phi["msigma_mat"];
  Eigen::MatrixXd phit_phi = mu_phi.transpose() * mu_phi;

  Eigen::MatrixXd mu_gamma = param_gamma["mu"];
  Eigen::VectorXd mu_tau = param_tau["mu"];
  Eigen::MatrixXd delta1_t = param_theta["delta1_t"];
  Eigen::MatrixXd delta2_t = param_theta["delta2_t"];

  // summing covariances of estimated phis
  Eigen::MatrixXd sum_msigma_phi = Eigen::MatrixXd::Constant(K, K, 0.0);
  for(int n = 0; n < S; n++)
  {
    sum_msigma_phi = sum_msigma_phi + msigma_phi.block(n * K, 0, K, K);
  }

  Eigen::VectorXd temp_vec = Eigen::VectorXd::Constant(K, 1.0);
  Eigen::MatrixXd Dhat_m = Eigen::MatrixXd::Constant(K, K, 1.0);

  for(int m = 0; m < M; m++)
  {
    temp_vec = mu_lambda.array() * mu_gamma.row(m).transpose().array();
    Dhat_m = temp_vec.asDiagonal();
    delta1_t.row(m) = (N * mu_tau(m)) / S * mu_phi.transpose() *
      E_hat.col(m);
    delta2_t.block(m * K, 0, K, K) = -1.0 / 2.0 / S * (
      N * mu_tau(m) * (phit_phi + sum_msigma_phi) + S * Dhat_m
    );
  }
  param_theta["delta1_t"] = delta1_t;
  param_theta["delta2_t"] = delta2_t;
  return(param_theta);
}

// [[Rcpp::export]]
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

// [[Rcpp::export]]
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

// [[Rcpp::export]]
Rcpp::List mvlm_vi_tau (
  Eigen::MatrixXd& E_hat, Eigen::MatrixXd& X, Rcpp::List& param_b0,
  Rcpp::List& param_b, Rcpp::List& param_phi, Rcpp::List& param_theta,
  int& N, int& M, int& P, int& S, int& K, double& a_tau, double& b_tau,
  Rcpp::List& param_tau
){
  // values to use throughout
  Eigen::VectorXd mu_b0 = param_b0["mu"];
  Eigen::VectorXd vsigma2_b0 = param_b0["vsigma2"];
  Eigen::MatrixXd mu_b = param_b["mu"];
  Eigen::MatrixXd msigma_b = param_b["msigma_mat"];
  Eigen::MatrixXd mu_phi = param_phi["mu"];
  Eigen::MatrixXd msigma_phi = param_phi["msigma_mat"];
  Eigen::MatrixXd mu_theta = param_theta["mu"];
  Eigen::MatrixXd msigma_theta = param_theta["msigma_mat"];
  Eigen::MatrixXd XtX = X.transpose() * X;
  Eigen::MatrixXd phit_phi = mu_phi.transpose() * mu_phi;

  // param_vals
  Eigen::VectorXd delta1_t = param_tau["delta1_t"];
  Eigen::VectorXd delta2_t = param_tau["delta2_t"];

  // summing covariances of estimated phis
  Eigen::MatrixXd sum_msigma_phi = Eigen::MatrixXd::Constant(K, K, 0.0);
  for(int n = 0; n < S; n++)
  {
    sum_msigma_phi = sum_msigma_phi + msigma_phi.block(n * K, 0, K, K);
  }

  // update params
  for(int m = 0; m < M; m++)
  {
    double bstar_m =
      E_hat.col(m).array().square().sum() +
      S * vsigma2_b0(m) +
      (msigma_b.block(m * P, 0, P, P) * XtX).trace() + (
        msigma_theta.block(m * K, 0, K, K) * (phit_phi + sum_msigma_phi)
      ).trace() +
      mu_theta.row(m) * sum_msigma_phi * mu_theta.row(m).transpose();

    delta1_t(m) = N / 2.0 + a_tau - 1;
    delta2_t(m) = -b_tau - N / 2.0 * 1.0 / S * bstar_m;
  }

  param_tau["delta1_t"] = delta1_t;
  param_tau["delta2_t"] = delta2_t;
  return(param_tau);
}

// [[Rcpp::export]]
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

// [[Rcpp::export]]
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

//**********************************************************************//
// cpp function to udpate z and psi together
//**********************************************************************//
// [[Rcpp::export]]
Rcpp::List mv_probit_vi_z_psi(
  Eigen::MatrixXi& Y_s, Eigen::MatrixXd& E_hat, Rcpp::List& param_theta,
  int& S, int& M, int& K, Rcpp::List& param_psi, Rcpp::List& param_z
){
  Eigen::VectorXd mu_theta = param_theta["mu"];
  Eigen::MatrixXd msigma_theta = param_theta["msigma_mat"];
  Eigen::MatrixXd mu_z = param_z["mu"];
  Eigen::MatrixXd mu_psi = param_psi["mu"];
  Eigen::MatrixXd msigma_mat_psi = param_psi["msigma_mat"];

  Eigen::MatrixXd G = mu_theta.transpose() * mu_theta +
    Eigen::MatrixXd::Identity(K, K);

  // summing covariances of estimated thetas
  for(int m = 0; m < M; m++)
  {
    G = G + msigma_theta.block(m * K, 0, K, K);
  }

  Eigen::LLT<Eigen::MatrixXd> chol_G(G);
  Eigen::VectorXd g(K);
  Eigen::VectorXd this_psi(K);
  Eigen::VectorXd psi_mu(K);
  Eigen::VectorXd this_z(M);
  Eigen::VectorXd z_mu(M);
  Eigen::MatrixXd ex_psi_sq(K, K);
  Eigen::VectorXd eta(M);
  Eigen::VectorXd ehat_s(M);
  Eigen::VectorXi y(M);

  int n_iter = 1000;
  Eigen::MatrixXd z_mat(n_iter, S * M);

  for(int s = 0; s < S; s++)
  {
    Rcpp::checkUserInterrupt();
    this_psi = Eigen::VectorXd::Constant(K, 0.0);
    this_z = Eigen::VectorXd::Constant(M, 0.0);
    psi_mu = Eigen::VectorXd::Constant(K, 0.0);
    z_mu = Eigen::VectorXd::Constant(M, 0.0);
    y = Y_s.row(s).transpose();
    ehat_s = E_hat.row(s).transpose();

    for(int i = 0; i < n_iter; i++)
    {
      eta = ehat_s + mu_theta * this_psi;
      probit_gibbs_z(y, eta, M, this_z);

      g = mu_theta.transpose() * (ehat_s + this_z);
      this_psi = chol_G.solve(g) +
        chol_G.matrixU().solve(conv(Rcpp::rnorm(K, 0, 1)));

      z_mu = z_mu + 1.0 / n_iter * this_z;
      psi_mu = psi_mu + 1.0 / n_iter * this_psi;

      z_mat.block(i, s * M, 1, M) = this_z;
    }

    mu_z.row(s) = z_mu.transpose();
    mu_psi.row(s) = psi_mu.transpose();
    msigma_mat_psi.block(s * K, 0, K, K) = chol_G.solve(Eigen::MatrixXd::Identity(K, K));
    Rcpp::Rcout << "Done with iteration " << s << "\r" << std::endl;
  }

  param_z["mu"] = mu_z;
  param_psi["mu"] = mu_psi;
  param_psi["msigma_mat"] = msigma_mat_psi;

  Rcpp::List retl;
  retl["param_z"] = param_z;
  retl["param_psi"] = param_psi;
  retl["z_mat"] = z_mat;

  return(retl);
}

