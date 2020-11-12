#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include "typedefs.h"
#include "include/helpers.hpp"
#include "include/multivariate_helpers.hpp"

using namespace Eigen;
using namespace Rcpp;
using namespace std;

// **********************************************************************
// calculate elbo for multivariate probit model
// **********************************************************************
double mv_lm_uninf_elbo(
  Eigen::MatrixXd& X_s, Eigen::MatrixXd& Y_s,
  Eigen::MatrixXd& mu_theta, Eigen::MatrixXd& msigma_theta,
  Eigen::MatrixXd& mu_psi, Eigen::MatrixXd& msigma_psi,
  Eigen::VectorXd& mu_b0, Eigen::VectorXd& vsigma2_b0,
  Eigen::MatrixXd& mu_B, Eigen::MatrixXd& msigma_B,
  double& logdet_msigma_psi, double& logdet_msigma_theta,
  double& logdet_msigma_B, double& astar_tau, double& bstar_tau, double& mu_tau,
  double& a_tau, double& b_tau, int& N, int& S, int& M, int& P, int& K
){

  Eigen::MatrixXd mu_prec;
  double mu_logdet_prec;
  Eigen::VectorXd one_S = Eigen::VectorXd::Constant(S, 1.0);

  // ********** ll **********
  Eigen::MatrixXd E_hat = Y_s - one_S * mu_b0.transpose() - X_s * mu_B.transpose() - 
    mu_psi * mu_theta.transpose();
  
  double ll =
    -N/2.0 * M * std::log(2.0 * M_PI) +
    N/2.0 * M * (Rf_digamma(astar_tau) - std::log(bstar_tau)) -
    mu_tau/2.0 * (
      E_hat.array().square().sum() +
      N * vsigma2_b0.array().sum() +
      M * N * 1.0 / S * (msigma_B * X_s.transpose() * X_s).trace() +
      N * M * (msigma_theta * msigma_psi).trace() +
      M * N * 1.0 / S * (msigma_theta * mu_psi.transpose() * mu_psi).trace() +
      N * (msigma_psi * mu_theta.transpose() * mu_theta).trace()
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
  
  // ----- tau -----
  double lp_tau =
    a_tau * std::log(b_tau) - Rf_lgammafn(a_tau) +
    (a_tau - 1.0) * (Rf_digamma(astar_tau) - std::log(bstar_tau)) -
    b_tau * mu_tau;

  // ********** lq **********
  double logdet_msigma_b0 = vsigma2_b0.array().log().sum();
  double lq_b0 = lq_mv_normal2(logdet_msigma_b0, M);
  double lq_psi = N * lq_mv_normal2(logdet_msigma_psi, K);
  double lq_theta = M * lq_mv_normal2(logdet_msigma_theta, K);
  double lq_B = M * lq_mv_normal2(logdet_msigma_B, P);
  double lq_tau =
    -Rf_lgammafn(astar_tau) + std::log(bstar_tau) +
    (astar_tau - 1.0) * Rf_digamma(astar_tau) - astar_tau;

  double elbo = ll + lp_psi + lp_theta + lp_b0 + lp_B + lp_tau - lq_psi -
    lq_theta - lq_b0 - lq_B - lq_tau;

  return(elbo);
}

// **********************************************************************
// CAVI
// **********************************************************************
//' Estimate the parameters in a multivariate linear model with the CAVI
//' algorithm.
//' @title Multivariate linear regression with a factor model - CAVI
//' @param Y matrix of responses
//' @param X matrix of predictors to control for
//' @param K number of factors in the factor model
//' @param n_iter number of iterations to run the Gibbs sampler
//' @param verbose True or False. Print status of the sampler.
//' @param a_tau Prior shape for the nugget term.
//' @param b_tau Prior rate for the nugget term.
//' @param rel_tol Relative tolerance for stopping
//' @param abs_tol Absolute tolerance for stopping; considered only after
//'   relative tolerance is met.
//' @export
// [[Rcpp::export]]
Rcpp::List mv_lm_uninf_cavi(
  Eigen::MatrixXd Y, Eigen::MatrixXd X, int K, int n_iter, bool verbose = true,
  double a_tau = 0.1, double b_tau = 0.1, double rel_tol = 0.0001,
  double abs_tol = 0.1
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

  Eigen::MatrixXd E_hat = Eigen::MatrixXd::Constant(N, M, 0.0);
  Eigen::MatrixXd diag_K = Eigen::MatrixXd::Identity(K, K);
  double mu_tau = 1.0;
  double this_sd = 1.0;
  double this_mu = 0.0;
  double astar_tau = 0.0;
  double bstar_tau = 0.0;
  int this_y;

  Eigen::VectorXd elbo = Eigen::VectorXd::Constant(n_iter, 1.0);

  double logdet_msigma_psi, logdet_msigma_B, logdet_msigma_theta;

  int iters = 0;
  for(int i = 0; i < n_iter; i++)
  {
    Rcpp::checkUserInterrupt();
    if(verbose && (i % 10 == 0)) {
      Rcpp::Rcout << "Done with Iteration " << i << " of " << n_iter << "\r";
    }

    // update psi
    E_hat = Y - one_N * mu_b0.transpose() - X * mu_B.transpose();
    msigma_psi_inv = diag_K + mu_tau * (
      M * msigma_theta + mu_theta.transpose() * mu_theta
    );

    Eigen::LLT<Eigen::MatrixXd> chol_psi_inv(msigma_psi_inv);
    msigma_psi = chol_psi_inv.solve(Eigen::MatrixXd::Identity(K, K));
    logdet_msigma_psi = -2.0 *
      chol_psi_inv.matrixL().toDenseMatrix().diagonal().array().log().sum();

    mu_psi = mu_tau * E_hat * mu_theta * msigma_psi;

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
    E_hat = Y - X * mu_B.transpose() - mu_psi * mu_theta.transpose();
    for(int m = 0; m < M; m++)
    {
      vsigma2_b0(m) = 1.0 / (mu_tau * N + 0.000001);
      mu_b0(m) = vsigma2_b0(m) * mu_tau * E_hat.col(m).array().sum();
    }

    // update B
    E_hat = Y - one_N * mu_b0.transpose() - mu_psi * mu_theta.transpose();
    msigma_B_inv = mu_tau * X.transpose() * X +
      0.000001 * Eigen::MatrixXd::Identity(P, P);

    Eigen::LLT<Eigen::MatrixXd> chol_B_inv(msigma_B_inv);
    msigma_B = chol_B_inv.solve(Eigen::MatrixXd::Identity(P, P));
    logdet_msigma_B = -2.0 *
      chol_B_inv.matrixL().toDenseMatrix().diagonal().array().log().sum();

    mu_B = mu_tau * E_hat.transpose() * X * msigma_B;

    // ----- tau -----
    E_hat = Y - one_N * mu_b0.transpose() - X * mu_B.transpose() -
      mu_psi * mu_theta.transpose();
    astar_tau = N/2.0 * M + a_tau;

    bstar_tau = b_tau + N/2.0 / S * (
      E_hat.array().square().sum() +
      N * vsigma2_b0.array().sum() +
      M * (msigma_B * X.transpose() * X).trace() +
      N * M * (msigma_theta * msigma_psi).trace() +
      M * (msigma_theta * mu_psi.transpose() * mu_psi).trace() +
      N * (msigma_psi * mu_theta.transpose() * mu_theta).trace()
    );
    mu_tau = astar_tau / bstar_tau;

    // elbo
    elbo(i) = mv_lm_uninf_elbo(
      X, Y, mu_theta, msigma_theta, mu_psi, msigma_psi, mu_b0, vsigma2_b0,
      mu_B, msigma_B, logdet_msigma_psi, logdet_msigma_theta,
      logdet_msigma_B, astar_tau, bstar_tau, mu_tau, a_tau, b_tau, N, S, M, P, K
    );

    iters = iters + 1;
    if(i > 4)
    {
      // check if lower bound decreases
      if(elbo(i) < elbo(i - 1))
        std::cout << "LOWER BOUND DECREASES" << "\n";
      if((1.0 - elbo(i) / elbo(i - 5)) < rel_tol) {
        if(std::abs(elbo(i) - elbo(i-5)) < abs_tol) {
          break;
        }
      }

      if(i == (n_iter - 1))
        std::cout << "VB DID NOT CONVERGE" << "\n";
    }
  }

  Rcpp::List retl;

  Rcpp::List b0;
  b0["dist"] = "independent multivariate normal";
  b0["mu"] = mu_b0;
  b0["vsigma2"] = vsigma2_b0;

  Rcpp::List B;
  B["dist"] = "matrix normal. independent over rows and each row has the same var-cov matrix";
  B["mu"] = mu_B;
  B["msigma"] = msigma_B;

  Rcpp::List theta;
  theta["dist"] = "matrix normal. independent over rows and each row has the same var-cov matrix";
  theta["mu"] = mu_theta;
  theta["msigma"] = msigma_theta;

  Rcpp::List tau;
  tau["dist"] = "univariate gamma";
  tau["shape"] = astar_tau;
  tau["rate"] = bstar_tau;

  retl["b0"] = b0;
  retl["B"] = B;
  retl["theta"] = theta;
  retl["tau"] = tau;
  retl["elbo"] = elbo.topRows(iters);

  return(retl);
}

// **********************************************************************
// SVI
// **********************************************************************
//' Run a SVI algorithm for the multivariate linear regression model.
//' @title Multivariate linear regression with a factor model
//' @param Y matrix of responses
//' @param X matrix of predictors to control for
//' @param K number of factors in the factor model
//' @param n_iter number of iterations to run the Gibbs sampler
//' @param verbose True or False. Print status of the sampler.
//' @param a_tau prior shape parameter for precisions
//' @param b_tau prior shape parameter for precisions
//' @param batch_size Size of the subsamples used to update the parameters.
//' @param const_rhot Used to set a constant step size in the gradient descent
//'   algorithm. If this parameter is greater than zero, it overrides the step
//'   size iterations calculated using kappa and omega.
//' @param omega Delay for the stepsize (\eqn{\omega}) for the gradient step.
//'   Interacts with \eqn{\kappa} via the formula \eqn{\rho_{t} = (t +
//'   \omega)^{-\kappa}}. This parameter has to be greater than or equal to zero.
//' @param kappa Forgetting rate for the step size iterations; \eqn{\kappa \in
//'   (0.5, 1)}
//' @export
// [[Rcpp::export]]
Rcpp::List mv_lm_uninf_svi(
  Eigen::MatrixXd Y, Eigen::MatrixXd X, int K, int n_iter, bool verbose = true,
  double a_tau = 0.1, double b_tau = 0.1, int batch_size = 10,
  double omega = 15.0, double kappa = 0.6, double const_rhot = 0.01
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
  double mu_tau = 1.0;
  double astar_tau = 1.0;
  double bstar_tau = 1.0;

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
  double delta1_tau = N/2.0 * M + a_tau - 1.0;
  double delta2_tau = N/2.0 * M + a_tau - 1.0;

  Eigen::MatrixXd msigma_B_inv = Eigen::MatrixXd::Identity(P, P);
  Eigen::MatrixXd msigma_psi_inv = Eigen::MatrixXd::Identity(K, K);
  Eigen::MatrixXd msigma_theta_inv = Eigen::MatrixXd::Identity(K, K);

  Eigen::MatrixXd E_hat = Eigen::MatrixXd::Constant(S, M, 0.0);
  Eigen::MatrixXd diag_K = Eigen::MatrixXd::Identity(K, K);
  Eigen::MatrixXd diag_P = Eigen::MatrixXd::Identity(P, P);

  double logdet_msigma_psi = 1.0;
  double logdet_msigma_B = 1.0;
  double logdet_msigma_theta = 1.0;
  Eigen::VectorXd elbo = Eigen::VectorXd::Constant(n_iter, 1.0);

  Eigen::MatrixXd Y_s = Eigen::MatrixXd::Constant(S, M, 0);
  Eigen::MatrixXd X_s = Eigen::MatrixXd::Constant(S, P, 0.0);
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
    get_subsample_mvlm(Y, X, Y_s, X_s, S, N, the_sample);

    // ----- local parameters: update psi -----
    E_hat = Y_s - one_S * mu_b0.transpose() - X_s * mu_B.transpose();
    msigma_psi_inv = diag_K + mu_tau * (
      M * msigma_theta + mu_theta.transpose() * mu_theta
    );

    Eigen::LLT<Eigen::MatrixXd> chol_psi_inv(msigma_psi_inv);
    msigma_psi = chol_psi_inv.solve(Eigen::MatrixXd::Identity(K, K));
    logdet_msigma_psi = -2.0 *
      chol_psi_inv.matrixL().toDenseMatrix().diagonal().array().log().sum();

    mu_psi = mu_tau * E_hat * mu_theta * msigma_psi;

    // ------------------------------------------------------------------------
    // global parameter updates; update the natural parameters first
    // ------------------------------------------------------------------------

    // update theta
    E_hat = Y_s - one_S * mu_b0.transpose() - X_s * mu_B.transpose();
    delta1_theta = (1.0 - rhot) * delta1_theta + rhot * (
      N * mu_tau / S * E_hat.transpose() * mu_psi
    );
    delta2_theta = (1.0 - rhot) * delta2_theta + rhot * (
      -1.0 / 2.0 / S * (
        N * mu_tau * (S * msigma_psi + mu_psi.transpose() * mu_psi) +
        S * 0.000001 * diag_K
      )
    );

    // udpate b0
    E_hat = Y_s - X_s * mu_B.transpose() - mu_psi * mu_theta.transpose();
    delta1_b0 = (1.0 - rhot) * delta1_b0 + rhot * (
      N * mu_tau / S * E_hat.colwise().sum().transpose()
    );
    delta2_b0 = (1.0 - rhot) * delta2_b0 + rhot * (
      -1.0 / 2.0 * Eigen::VectorXd::Constant(M, 1.0) * (mu_tau * N + 0.000001)
    );

    // update B
    E_hat = Y_s - one_S * mu_b0.transpose() - mu_psi * mu_theta.transpose();
    delta1_B = (1.0 - rhot) * delta1_B + rhot * (
      N * mu_tau / S * E_hat.transpose() * X_s
    );
    delta2_B = (1.0 - rhot) * delta2_B + rhot * (
      -1.0 / 2.0 / S * (
        N * mu_tau * X_s.transpose() * X_s + S * 0.000001 * diag_P
      )
    );

    // ----- tau -----
    E_hat = Y_s - one_S * mu_b0.transpose() - X_s * mu_B.transpose() -
      mu_psi * mu_theta.transpose();
    delta1_tau = N/2.0 * M + a_tau - 1.0;

    delta2_tau = -b_tau - N/2.0 / S * (
      E_hat.array().square().sum() +
      N * vsigma2_b0.array().sum() +
      M * (msigma_B * X.transpose() * X).trace() +
      N * M * (msigma_theta * msigma_psi).trace() +
      M * (msigma_theta * mu_psi.transpose() * mu_psi).trace() +
      N * (msigma_psi * mu_theta.transpose() * mu_theta).trace()
    );

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

    // B
    Eigen::LLT<Eigen::MatrixXd> chol_B_delta2(-2.0 * delta2_B);
    msigma_B = chol_B_delta2.solve(Eigen::MatrixXd::Identity(P, P));
    mu_B = delta1_B * msigma_B;
    logdet_msigma_B = -2.0 *
      chol_B_delta2.matrixL().toDenseMatrix().diagonal().array().log().sum();

    // b0
    vsigma2_b0 = -1.0 / 2.0 * delta2_b0.array().inverse();
    mu_b0 = vsigma2_b0.array() * delta1_b0.array();

    // tau
    astar_tau = delta1_tau + 1.0;
    bstar_tau = -delta2_tau;
    mu_tau = astar_tau / bstar_tau;

    // elbo
    elbo(i) = mv_lm_uninf_elbo(
      X_s, Y_s, mu_theta, msigma_theta, mu_psi, msigma_psi, mu_b0, vsigma2_b0,
      mu_B, msigma_B, logdet_msigma_psi, logdet_msigma_theta,
      logdet_msigma_B, astar_tau, bstar_tau, mu_tau, a_tau, b_tau, N, S, M, P, K
    );
  }

  Rcpp::List retl;

  Rcpp::List b0;
  b0["dist"] = "independent multivariate normal";
  b0["mu"] = mu_b0;
  b0["vsigma2"] = vsigma2_b0;

  Rcpp::List B;
  B["dist"] = "matrix normal. independent over rows and each row has the same var-cov matrix";
  B["mu"] = mu_B;
  B["msigma"] = msigma_B;

  Rcpp::List theta;
  theta["dist"] = "matrix normal. independent over rows and each row has the same var-cov matrix";
  theta["mu"] = mu_theta;
  theta["msigma"] = msigma_theta;

  Rcpp::List tau;
  tau["dist"] = "univariate gamma";
  tau["shape"] = astar_tau;
  tau["rate"] = bstar_tau;

  retl["b0"] = b0;
  retl["B"] = B;
  retl["theta"] = theta;
  retl["tau"] = tau;
  retl["elbo"] = elbo;

  return(retl);
}
