#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include "typedefs.h"
#include "include/helpers.hpp"
#include "include/multivariate_helpers.hpp"

using namespace Rcpp;
using namespace std;
using namespace Eigen;

// **********************************************************************
// calculate elbo for multivariate probit model
// **********************************************************************
double mv_probit_uninf_elbo(
  Eigen::MatrixXd& X_s, Eigen::MatrixXi& Y_s,
  Eigen::MatrixXd& mu_theta, Eigen::MatrixXd& msigma_theta,
  Eigen::MatrixXd& mu_psi, Eigen::MatrixXd& msigma_psi,
  Eigen::VectorXd& mu_b0, Eigen::VectorXd& vsigma2_b0,
  Eigen::MatrixXd& mu_B, Eigen::MatrixXd& msigma_B,
  double& logdet_msigma_psi, double& logdet_msigma_theta,
  double& logdet_msigma_B, int& N, int& S, int& M, int& P, int& K
){

  Eigen::MatrixXd mu_prec;
  double mu_logdet_prec;

  // ********** Z **********
  double lp_m_lq_z = mv_probit_lp_m_lq_z(
    X_s, Y_s, mu_theta, msigma_theta, mu_psi, msigma_psi, mu_b0, vsigma2_b0,
    mu_B, msigma_B, logdet_msigma_psi, logdet_msigma_theta, N, S, M
  );

  // ********** lp ***********
  // ------ psi -----
  mu_prec = Eigen::MatrixXd::Identity(K, K);
  mu_logdet_prec = 0.0;

  double lp_psi = lp_indep_matrix_normal(
    mu_prec, mu_logdet_prec, mu_psi, msigma_psi, K, S
  );

  lp_psi = N * lp_psi / S;

  // ------ theta -----
  mu_prec = 0.000001 * Eigen::MatrixXd::Identity(K, K);
  mu_logdet_prec = -6.0 * K * std::log(10.0);

  double lp_theta = lp_indep_matrix_normal(
    mu_prec, mu_logdet_prec, mu_theta, msigma_theta, K, M
  );

  // ----- b0 -----
  double mu_prec_b0, mu_log_prec, lp_b0, mu, sigma2;

  mu_prec_b0 = 0.000001;
  mu_log_prec = -6.0 * std::log(10.0);
  lp_b0 = 0.0;

  for(int m = 0; m < M; m++)
  {
    mu = mu_b0(m);
    sigma2 = vsigma2_b0(m);

    lp_b0 +=
      -1.0 / 2.0 * std::log(2.0 * M_PI) +
      1.0 / 2.0 * mu_log_prec -
      mu_prec_b0 / 2.0 * (mu * mu + sigma2);
  }

  // ------ B -----
  Eigen::MatrixXd mu_prec_B = 0.000001 * Eigen::MatrixXd::Identity(P, P);
  mu_logdet_prec = -6.0 * P * std::log(10.0);

  double lp_B = lp_indep_matrix_normal(
    mu_prec_B, mu_logdet_prec, mu_B, msigma_B, P, M
  );

  // ********** lq **********
  double logdet_msigma_b0 = vsigma2_b0.array().log().sum();
  double lq_b0 = lq_mv_normal2(logdet_msigma_b0, M);
  double lq_psi = N * lq_mv_normal2(logdet_msigma_psi, K);
  double lq_theta = M * lq_mv_normal2(logdet_msigma_theta, K);
  double lq_B = M * lq_mv_normal2(logdet_msigma_B, P);

  double elbo = lp_m_lq_z + lp_psi + lp_theta + lp_b0 + lp_B - lq_psi -
    lq_theta - lq_b0 - lq_B;

  return(elbo);
}

// **********************************************************************
// CAVI
// **********************************************************************
// [[Rcpp::export]]
Rcpp::List mv_probit_uninf_cavi(
  Eigen::MatrixXi Y, Eigen::MatrixXd X, int K, int n_iter, bool verbose = true,
  double a_tau = 0.1, double b_tau = 0.1, double rel_tol = 0.00001
){
  // problem info
  int N = Y.rows();
  int M = Y.cols();
  int P = X.cols();
  int S = N;
  Eigen::VectorXd one_N = Eigen::VectorXd::Constant(N, 1.0);

  // initializae matricies
  // mean storage
  Eigen::VectorXd mu_b0 = 0.1 * Eigen::VectorXd::Random(M);
  Eigen::MatrixXd mu_B = 0.1 * Eigen::MatrixXd::Random(M, P);
  Eigen::MatrixXd mu_theta = 0.1 * Eigen::MatrixXd::Random(M, K);
  Eigen::MatrixXd mu_psi = 0.1 * Eigen::MatrixXd::Random(N, K);

  // variance storage
  Eigen::VectorXd vsigma2_b0 = Eigen::VectorXd::Constant(M, 1.0);
  Eigen::MatrixXd msigma_B = Eigen::MatrixXd::Identity(P, P);
  Eigen::MatrixXd msigma_psi = Eigen::MatrixXd::Identity(K, K);
  Eigen::MatrixXd msigma_theta = Eigen::MatrixXd::Identity(K, K);

  Eigen::MatrixXd msigma_B_inv = Eigen::MatrixXd::Identity(P, P);
  Eigen::MatrixXd msigma_psi_inv = Eigen::MatrixXd::Identity(K, K);
  Eigen::MatrixXd msigma_theta_inv = Eigen::MatrixXd::Identity(K, K);

  Eigen::MatrixXd mu_Z = Eigen::MatrixXd::Constant(N, M, 0.0);
  Eigen::MatrixXd msigma2_Z = Eigen::MatrixXd::Constant(N, M, 0.0);

  Eigen::MatrixXd E_hat = Eigen::MatrixXd::Constant(N, M, 0.0);
  Eigen::MatrixXd diag_K = Eigen::MatrixXd::Identity(K, K);
  double mu_tau = 1.0;
  double this_sd = 1.0;
  double this_mu = 0.0;
  int this_y;

  Eigen::VectorXd elbo = Eigen::VectorXd::Constant(n_iter, 1.0);

  double logdet_msigma_psi, logdet_msigma_B, logdet_msigma_theta;

  // initialize Z
  mu_Z = Eigen::MatrixXd::Constant(N, M, 0.0);
  for(int n = 0; n < N; n++)
  {
    for(int m = 0; m < M; m++)
    {
      this_mu = mu_Z(n, m);
      this_sd = 1.0;
      this_y = Y(n, m);
      mu_var_trunc_norm(this_y, this_mu, this_sd);
      mu_Z(n, m) = this_mu;
      msigma2_Z(n, m) = this_sd;
    }
  }

  int iters = 0;
  for(int i = 0; i < n_iter; i++)
  {
    Rcpp::checkUserInterrupt();
    if(verbose && (i % 10 == 0)) {
      Rcpp::Rcout << "Done with Iteration " << i << " of " << n_iter << "\r";
    }

    // update psi
    E_hat = mu_Z - one_N * mu_b0.transpose() - X * mu_B.transpose();
    msigma_psi_inv = diag_K + mu_tau * (
      M * msigma_theta + mu_theta.transpose() * mu_theta
    );

    Eigen::LLT<Eigen::MatrixXd> chol_psi_inv(msigma_psi_inv);
    msigma_psi = chol_psi_inv.solve(Eigen::MatrixXd::Identity(K, K));
    logdet_msigma_psi = -2.0 *
      chol_psi_inv.matrixL().toDenseMatrix().diagonal().array().log().sum();

    mu_psi = mu_tau * E_hat * mu_theta * msigma_psi;

    // update Z
    mu_Z = Eigen::MatrixXd::Constant(N, M, 0.0);
    for(int n = 0; n < N; n++)
    {
      for(int m = 0; m < M; m++)
      {
        this_mu = mu_Z(n, m);
        this_sd = 1.0 / std::sqrt(mu_tau);
        this_y = Y(n, m);
        mu_var_trunc_norm(this_y, this_mu, this_sd);

        // Rcpp::Rcout << "Here 1: " << std::endl;

        mu_Z(n, m) = this_mu;
        msigma2_Z(n, m) = this_sd;
      }
    }

    // update theta
    msigma_theta_inv = 0.000001 * diag_K + mu_tau * (
      N * msigma_psi + mu_psi.transpose() * mu_psi
    );

    Eigen::LLT<Eigen::MatrixXd> chol_theta_inv(msigma_theta_inv);
    msigma_theta = chol_theta_inv.solve(Eigen::MatrixXd::Identity(K, K));
    logdet_msigma_theta = -2.0 *
      chol_theta_inv.matrixL().toDenseMatrix().diagonal().array().log().sum();

    mu_theta = mu_tau * E_hat.transpose() * mu_psi * msigma_theta;

    // udpate b0
    E_hat = mu_Z - X * mu_B.transpose() - mu_psi * mu_theta.transpose();
    for(int m = 0; m < M; m++)
    {
      vsigma2_b0(m) = 1.0 / (mu_tau * N + 0.000001);
      mu_b0(m) = vsigma2_b0(m) * mu_tau * E_hat.col(m).array().sum();
    }

    // update B
    E_hat = mu_Z - one_N * mu_b0.transpose() - mu_psi * mu_theta.transpose();
    msigma_B_inv = mu_tau * X.transpose() * X +
      0.000001 * Eigen::MatrixXd::Identity(P, P);

    Eigen::LLT<Eigen::MatrixXd> chol_B_inv(msigma_B_inv);
    msigma_B = chol_B_inv.solve(Eigen::MatrixXd::Identity(P, P));
    logdet_msigma_B = -2.0 *
      chol_B_inv.matrixL().toDenseMatrix().diagonal().array().log().sum();

    mu_B = mu_tau * E_hat.transpose() * X * msigma_B;

    elbo(i) = mv_probit_uninf_elbo(
      X, Y, mu_theta, msigma_theta, mu_psi, msigma_psi, mu_b0, vsigma2_b0,
      mu_B, msigma_B, logdet_msigma_psi, logdet_msigma_theta,
      logdet_msigma_B, N, S, M, P, K
    );

    iters = iters + 1;
    if(i > 4)
    {
      // check if lower bound decreases
      if(elbo(i) < elbo(i - 1))
        std::cout << "LOWER BOUND DECREASES" << "\n";
      if((1.0 - elbo(i) / elbo(i - 5)) < rel_tol) {
        if(elbo(i) - elbo(i-5) < 0.1) {
          break;
        }
      }

      if(i == (n_iter - 1))
        std::cout << "VB DID NOT CONVERGE" << "\n";
    }
  }

  Rcpp::List retl;

  retl["mu_b0"] = mu_b0;
  retl["mu_B"] = mu_B;
  retl["mu_theta"] = mu_theta;
  retl["vsigma2_b0"] = vsigma2_b0;
  retl["msigma_B"] = msigma_B;
  retl["msigma_theta"] = msigma_theta;
  retl["elbo"] = elbo.topRows(iters);

  return(retl);
}

// **********************************************************************
// SVI
// **********************************************************************
// [[Rcpp::export]]
Rcpp::List mv_probit_uninf_svi(
  Eigen::MatrixXi Y, Eigen::MatrixXd X, int K, int n_iter, bool verbose = true,
  int batch_size = 10, double omega = 15.0, double kappa = 0.6,
  double const_rhot = 0.01
){
  // problem info
  int N = Y.rows();
  int M = Y.cols();
  int P = X.cols();
  int S = batch_size;
  Eigen::VectorXd one_S = Eigen::VectorXd::Constant(S, 1.0);

  // initialize matricies
  // mean storage
  Eigen::MatrixXd mu_Z = Eigen::MatrixXd::Constant(S, M, 0.0);
  Eigen::VectorXd mu_b0 = 0.2 * Eigen::VectorXd::Random(M);
  Eigen::MatrixXd mu_B = 0.2 * Eigen::MatrixXd::Random(M, P);
  Eigen::MatrixXd mu_theta = 0.3 * Eigen::MatrixXd::Random(M, K);
  Eigen::MatrixXd mu_psi = 0.1 * Eigen::MatrixXd::Random(S, K);

  // variance storage
  Eigen::VectorXd vsigma2_b0 = Eigen::VectorXd::Constant(M, 1.0);
  Eigen::MatrixXd msigma_B = Eigen::MatrixXd::Identity(P, P);
  Eigen::MatrixXd msigma_psi = Eigen::MatrixXd::Identity(K, K);
  Eigen::MatrixXd msigma_theta = Eigen::MatrixXd::Identity(K, K);

  // natural parameters
  Eigen::VectorXd delta1_b0 = mu_b0;
  Eigen::VectorXd delta2_b0 = Eigen::VectorXd::Constant(M, -0.5);
  Eigen::MatrixXd delta1_B = mu_B;
  Eigen::MatrixXd delta2_B = -1.0 / 2.0 * Eigen::MatrixXd::Identity(P, P);
  Eigen::MatrixXd delta1_theta = mu_theta;
  Eigen::MatrixXd delta2_theta = -1.0 / 2.0 * Eigen::MatrixXd::Identity(K, K);

  Eigen::MatrixXd msigma_B_inv = Eigen::MatrixXd::Identity(P, P);
  Eigen::MatrixXd msigma_psi_inv = Eigen::MatrixXd::Identity(K, K);
  Eigen::MatrixXd msigma_theta_inv = Eigen::MatrixXd::Identity(K, K);

  Eigen::MatrixXd E_hat = Eigen::MatrixXd::Constant(S, M, 0.0);
  Eigen::MatrixXd diag_K = Eigen::MatrixXd::Identity(K, K);
  Eigen::MatrixXd diag_P = Eigen::MatrixXd::Identity(P, P);
  double mu_tau = 1.0;
  double this_sd = 1.0;
  double this_mu = 0.0;
  int this_y;

  double logdet_msigma_psi = 1.0;
  double logdet_msigma_B = 1.0;
  double logdet_msigma_theta = 1.0;
  Eigen::VectorXd elbo = Eigen::VectorXd::Constant(n_iter, 1.0);

  Eigen::MatrixXi Y_s = Eigen::MatrixXi::Constant(S, M, 0);
  Eigen::MatrixXd X_s = Eigen::MatrixXd::Constant(S, P, 0.0);
  Eigen::VectorXd local_elbo = Eigen::VectorXd::Constant(100, 1.0);
  Rcpp::IntegerVector the_sample = Rcpp::seq(0, S - 1);
  Rcpp::IntegerVector seq_samp = Rcpp::seq(0, N - 1);
  double rhot;

  int iters = 0;
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

    // ----- subsample the data -----
    the_sample = Rcpp::sample(seq_samp, S, false);
    get_subsample_mv_probit(Y, X, Y_s, X_s, S, N, the_sample);

    // Rcpp::Rcout << the_sample << std::endl;
    // Rcpp::Rcout << Y_s << std::endl;
    // Rcpp::Rcout << X_s << std::endl;

    // Rcpp::Rcout << "Breaks 0" << std::endl;

    // ------------------------------------------------------------------------
    // local parameters loop; run this until there is local convergence
    // ------------------------------------------------------------------------
    // this information about psi does not change in the loop
    mu_psi = Eigen::MatrixXd::Constant(S, K, 0.0);
    msigma_psi_inv = diag_K + M * msigma_theta + mu_theta.transpose() * mu_theta;
    Eigen::LLT<Eigen::MatrixXd> chol_psi_inv(msigma_psi_inv);
    msigma_psi = chol_psi_inv.solve(diag_K);
    logdet_msigma_psi = -2.0 *
      chol_psi_inv.matrixL().toDenseMatrix().diagonal().array().log().sum();

    // // loop until local convergence
    for(int j = 0; j < 100; j++)
    {
      Rcpp::checkUserInterrupt();

      //Rcpp::Rcout << "Breaks 2" << std::endl;
      // update Z
      mu_Z = one_S * mu_b0.transpose() + X_s * mu_B.transpose() +
        mu_psi * mu_theta.transpose();
      // Rcpp::Rcout << "mu_Z" << std::endl;
      // Rcpp::Rcout << mu_Z << std::endl;
      for(int s = 0; s < S; s++)
      {
        for(int m = 0; m < M; m++)
        {
          Rcpp::checkUserInterrupt();
          this_mu = mu_Z(s, m);
          this_y = Y_s(s, m);
          this_sd = 1.0;
          mu_var_trunc_norm(this_y, this_mu, this_sd);
          mu_Z(s, m) = this_mu;
        }
      }

      // update psi
      E_hat = mu_Z - one_S * mu_b0.transpose() - X_s * mu_B.transpose();
      mu_psi = E_hat * mu_theta * msigma_psi;

      // local elbo
      local_elbo(j) = mv_probit_uninf_elbo(
        X_s, Y_s, mu_theta, msigma_theta, mu_psi, msigma_psi, mu_b0, vsigma2_b0,
        mu_B, msigma_B, logdet_msigma_psi, logdet_msigma_theta, logdet_msigma_B,
        N, S, M, P, K
      );

      //Rcpp::Rcout << "Breaks 4" << std::endl;

      if(j > 5)
      {
        if((1.0 - local_elbo(j) / local_elbo(j - 5)) < 0.01)
        {
          //Rcpp::Rcout << "Broke internal loop after " << j << " iterations" <<
          //  std::endl;
          break;
        }
        if(j == 99) {
          Rcpp::Rcout << "Internal loop did not break" << std::endl;
        }
      }
    }

    // ------------------------------------------------------------------------
    // global parameter updates; update the natural parameters first
    // ------------------------------------------------------------------------

    // Rcpp::Rcout << "Breaks 5" << std::endl;
    // update theta
    E_hat = mu_Z - one_S * mu_b0.transpose() - X_s * mu_B.transpose();
    delta1_theta = (1.0 - rhot) * delta1_theta + rhot * (
      N * 1.0 / S * E_hat.transpose() * mu_psi
    );
    delta2_theta = (1.0 - rhot) * delta2_theta + rhot * (
      -1.0 / 2.0 / S * (
        N * (S * msigma_psi + mu_psi.transpose() * mu_psi) +
        S * 0.000001 * diag_K
      )
    );

    // Rcpp::Rcout << "Breaks 6" << std::endl;
    // udpate b0
    E_hat = mu_Z - X_s * mu_B.transpose() - mu_psi * mu_theta.transpose();
    delta1_b0 = (1.0 - rhot) * delta1_b0 + rhot * (
      N * 1.0 / S * E_hat.colwise().sum().transpose()
    );
    delta2_b0 = (1.0 - rhot) * delta2_b0 + rhot * (
      -1.0 / 2.0 * Eigen::VectorXd::Constant(M, 1.0) * (N + 0.000001)
    );

    // Rcpp::Rcout << "Breaks 7" << std::endl;
    // update B
    E_hat = mu_Z - one_S * mu_b0.transpose() - mu_psi * mu_theta.transpose();
    delta1_B = (1.0 - rhot) * delta1_B + rhot * (
      N * 1.0 / S * E_hat.transpose() * X_s
    );
    delta2_B = (1.0 - rhot) * delta2_B + rhot * (
      -1.0 / 2.0 / S * (N * X_s.transpose() * X_s + S * 0.000001 * diag_P)
    );

    // Rcpp::Rcout << delta2_theta << std::endl;
    // Rcpp::Rcout << msigma_theta << std::endl;

    // ------------------------------------------------------------------------
    // natural to canonical
    // ------------------------------------------------------------------------
    // Rcpp::Rcout << "Breaks 8" << std::endl;
    // theta
    Eigen::LLT<Eigen::MatrixXd> chol_theta_delta2(-2.0 * delta2_theta);
    msigma_theta = chol_theta_delta2.solve(Eigen::MatrixXd::Identity(K, K));
    mu_theta = delta1_theta * msigma_theta;
    logdet_msigma_theta = -2.0 *
      chol_theta_delta2.matrixL().toDenseMatrix().diagonal().array().log().sum();

    // Rcpp::Rcout << "Breaks 9" << std::endl;

    // B
    Eigen::LLT<Eigen::MatrixXd> chol_B_delta2(-2.0 * delta2_B);
    msigma_B = chol_B_delta2.solve(Eigen::MatrixXd::Identity(P, P));
    mu_B = delta1_B * msigma_B;
    logdet_msigma_B = -2.0 *
      chol_B_delta2.matrixL().toDenseMatrix().diagonal().array().log().sum();

    // b0
    vsigma2_b0 = -1.0 / 2.0 * delta2_b0.array().inverse();
    mu_b0 = vsigma2_b0.array() * delta1_b0.array();

    // Rcpp::Rcout << "Breaks 10" << std::endl;

    elbo(i) = mv_probit_uninf_elbo(
      X_s, Y_s, mu_theta, msigma_theta, mu_psi, msigma_psi, mu_b0, vsigma2_b0,
      mu_B, msigma_B, logdet_msigma_psi, logdet_msigma_theta,
      logdet_msigma_B, N, S, M, P, K
    );
  }

  Rcpp::List retl;

  retl["mu_b0"] = mu_b0;
  retl["mu_B"] = mu_B;
  retl["mu_theta"] = mu_theta;
  retl["vsigma2_b0"] = vsigma2_b0;
  retl["msigma_B"] = msigma_B;
  retl["msigma_theta"] = msigma_theta;
  retl["elbo"] = elbo;
  retl["local_elbo"] = local_elbo;

  return(retl);
}
