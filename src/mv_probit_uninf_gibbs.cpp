#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include "typedefs.h"
#include "include/helpers.hpp"
#include "include/multivariate_helpers.hpp"

using namespace Rcpp;
using namespace std;
using namespace Eigen;

//**********************************************************************//
// Main function
//**********************************************************************//
// Run a Gibbs sampler for the multivariate probit model. Note that the
// parameters are not normalized
// [[Rcpp::export]]
Rcpp::List mv_probit_uninf_gibbs(
    Eigen::MatrixXi Y, Eigen::MatrixXd X, int K = 2, int n_iter = 10000,
    int burn_in = 5000, bool verbose = true
){
  // problem info
  int N = X.rows();
  int P = X.cols();
  int M = Y.cols();
  Eigen::VectorXd one_N = Eigen::VectorXd::Constant(N, 1.0);

  int n_samps = n_iter - burn_in;

  // initializing matricies to store results
  Eigen::MatrixXd b0_mat = Eigen::MatrixXd::Constant(n_samps, M, 1.0);
  Eigen::MatrixXd B_mat = Eigen::MatrixXd::Constant(n_samps, M * P, 1.0);
  Eigen::MatrixXd mtheta_mat = Eigen::MatrixXd::Constant(n_samps, M * K, 1.0);
  Eigen::VectorXd tau_vec = Eigen::VectorXd::Constant(n_samps, 1.0);
  Eigen::VectorXd loglik_vec = Eigen::VectorXd::Constant(n_samps, 1.0);

  // starting values
  Eigen::VectorXd b0 = Eigen::VectorXd::Constant(M, 0.0);
  Eigen::MatrixXd B = Eigen::MatrixXd::Constant(M, P, 0.0);
  Eigen::MatrixXd mtheta = Eigen::MatrixXd::Constant(M, K, 0.0);
  Eigen::MatrixXd mpsi = Eigen::MatrixXd::Constant(N, K, 0.0);
  double tau = 1.0;

  // probit specific
  Eigen::MatrixXd Z = Eigen::MatrixXd::Constant(N, M, 0.0);
  Eigen::MatrixXd E_hat = Eigen::MatrixXd::Constant(N, M, 0.0);

  int iter = 0;
  for(int i = 0; i < n_iter; i++)
  {
    // check interrupt and print progress
    Rcpp::checkUserInterrupt();
    if(verbose && (i % 100 == 0)) {
      Rcout << "Done with Iteration " << i << " of " << n_iter << "\r";
    }

    // Z
    E_hat = one_N * b0.transpose() + X * B.transpose() +
      mpsi * mtheta.transpose();
    mv_probit_gibbs_Z(Y, E_hat, tau, N, M, Z);

    // mpsi
    E_hat = Z - one_N * b0.transpose() - X * B.transpose();
    mvlm_uninf_gibbs_mpsi(E_hat, mtheta, tau, N, K, mpsi);

    // mtheta
    mvlm_uninf_gibbs_mtheta(
      E_hat, mpsi, tau, M, K, mtheta
    );

    // b_0
    E_hat = Z - X * B.transpose() - mpsi * mtheta.transpose();
    mvlm_uninf_gibbs_b0(E_hat, tau, N, M, b0);

    // B
    E_hat = Z - one_N * b0.transpose() - mpsi * mtheta.transpose();
    mvlm_uninf_gibbs_B(E_hat, X, tau, M, P, B);

    // storing results; need maps for B, mpsi, mtheta, mgamma, and cov_mat
    Eigen::Map<VectorXd> B_vec(B.data(), B.size());
    Eigen::Map<VectorXd> theta_vec(mtheta.data(), mtheta.size());

    if(i >= burn_in)
    {
      b0_mat.row(iter) = b0;
      B_mat.row(iter) = B_vec;
      mtheta_mat.row(iter) = theta_vec;
      iter = iter + 1;
    }

    // log likelihood
    //E_hat = Z - one_N * b0.transpose() - X * B.transpose() -
    //  mpsi * mtheta.transpose();
    //loglik_vec(i) = 1.0/2.0 * tau.array().log().sum() -
    //  (D_tau * E_hat.transpose() * E_hat).trace();
  }

  Rcpp::List retl;
  retl["b0_mat"] = b0_mat;
  retl["B_mat"] = B_mat;
  retl["mtheta_mat"] = mtheta_mat;
  retl["loglik_vec"] = loglik_vec;
  return retl;
}
