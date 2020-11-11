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
  Eigen::MatrixXd E_hat = Y_s - one_S * mu_b0.transpose() -
    X_s * mu_B.transpose() - mu_psi * mu_theta.transpose();

  double ll = 
    -N/2.0 * M * std::log(2.0 * M_PI) + 
    N/2.0 * M * (Rf_digamma(astar_tau) - std::log(bstar_tau)) - 
    mu_tau/2.0 * E_hat.array().square().sum();

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
    (a_tau - 1) * (Rf_digamma(astar_tau) - std::log(bstar_tau)) -
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
// [[Rcpp::export]]
Rcpp::List mv_lm_uninf_cavi_cpp(
  Eigen::MatrixXd Y, Eigen::MatrixXd X, int K, int n_iter, bool verbose = true,
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
    
    bstar_tau = b_tau + 1.0/2.0 * (
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
  retl["vsigma2_b0"] = vsigma2_b0;
  retl["mu_B"] = mu_B;
  retl["msigma_B"] = msigma_B;
  retl["mu_theta"] = mu_theta;
  retl["msigma_theta"] = msigma_theta;
  retl["elbo"] = elbo.topRows(iters);

  return(retl);
}
