#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include "typedefs.h"
#include "include/helpers.hpp"
#include "include/multivariate_helpers.hpp"

using namespace Rcpp;
using namespace std;
using namespace Eigen;

//************************************************************************
// helpers
//************************************************************************
Eigen::VectorXd mv_lm_uninf_gibbs_b0(
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

Eigen::MatrixXd mv_lm_uninf_gibbs_B(
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

Eigen::MatrixXd mv_lm_uninf_gibbs_mtheta(
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

Eigen::MatrixXd mv_lm_uninf_gibbs_mphi(
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

Eigen::VectorXd mv_lm_uninf_gibbs_tau(
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

Eigen::MatrixXd mv_lm_uninf_gibbs_mgamma(
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

Eigen::VectorXd mv_lm_uninf_gibbs_xi(
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
// Main function
//**********************************************************************//
// [[Rcpp::export]]
Rcpp::List mv_lm_uninf_gibbs_cpp(
  Eigen::MatrixXd Y, Eigen::MatrixXd X, int K = 2, int n_iter = 10000,
  int burn_in = 5000, bool verbose = true, double a_tau = 0.1,
  double b_tau = 0.1
){
  // problem info
  int N = X.rows();
  int P = X.cols();
  int M = Y.cols();
  Eigen::VectorXd one_N = Eigen::VectorXd::Constant(N, 1.0);

  // scale X
  Eigen::RowVectorXd vmu_x = X.colwise().mean();
  Eigen::RowVectorXd vsigma_x =
      (X.rowwise() - vmu_x).colwise().squaredNorm() / (X.rows() - 1);
  vsigma_x = vsigma_x.array().sqrt();
  Eigen::VectorXd s_x = vmu_x.array() / vsigma_x.array();
  X = (X.rowwise() - vmu_x).array().rowwise() / vsigma_x.array();

  // initializing matricies to store results
  Eigen::MatrixXd b0_mat = Eigen::MatrixXd::Constant(n_iter, M, 1.0);
  Eigen::MatrixXd B_mat = Eigen::MatrixXd::Constant(n_iter, M * P, 1.0);
  Eigen::MatrixXd mtheta_mat = Eigen::MatrixXd::Constant(n_iter, M * K, 1.0);
  Eigen::MatrixXd mphi_mat = Eigen::MatrixXd::Constant(n_iter, N * K, 1.0);
  Eigen::MatrixXd tau_mat = Eigen::MatrixXd::Constant(n_iter, M, 1.0);
  Eigen::VectorXd loglik_vec = Eigen::VectorXd::Constant(n_iter, 1.0);

  // starting values
  Eigen::VectorXd b0 = Eigen::VectorXd::Constant(M, 0.0);
  Eigen::MatrixXd B = Eigen::MatrixXd::Constant(M, P, 0.0);
  Eigen::MatrixXd mtheta = Eigen::MatrixXd::Constant(M, K, 0.0);
  Eigen::MatrixXd mphi = Eigen::MatrixXd::Constant(N, K, 0.0);
  Eigen::VectorXd tau = Eigen::VectorXd::Constant(M, 1.0);
  Eigen::MatrixXd mgamma = Eigen::MatrixXd::Constant(M, K, 1.0);
  Eigen::VectorXd xi = Eigen::VectorXd::Constant(K, 1.0);
  for(int k = 0; k <= K; k++)
    xi(k) = 1.0 + k / 2.0;
  Eigen::VectorXd lambda = cum_prod(xi);
  double a1 = 2.1; double a2 = 10.1; double nu = 3;

  Eigen::MatrixXd E_hat = Eigen::MatrixXd::Constant(N, M, 0.0);
  Eigen::MatrixXd D_temp = Eigen::MatrixXd::Constant(M, M, 0.0);
  Eigen::MatrixXd cov_mat = Eigen::MatrixXd::Constant(M, M, 0.0);
  Eigen::VectorXd temp_b0 = Eigen::VectorXd::Constant(M, 0.0);
  Eigen::MatrixXd temp_B = Eigen::MatrixXd::Constant(M, P, 0.0);

  for(int i = 0; i < n_iter; i++)
  {
    // check interrupt and print progress
    Rcpp::checkUserInterrupt();
    if(verbose && (i % 100 == 0)) {
        Rcout << "Done with Iteration " << i << " of " << n_iter << "\r";
    }

    // mphi
    E_hat = Y - one_N * b0.transpose() - X * B.transpose();
    mv_lm_uninf_gibbs_mphi(E_hat, mtheta, tau, N, K, mphi);

    // mtheta
    mv_lm_uninf_gibbs_mtheta(
      E_hat, mphi, mgamma, tau, lambda, M, K, mtheta
    );

    // b_0
    E_hat = Y - X * B.transpose() - mphi * mtheta.transpose();
    mv_lm_uninf_gibbs_b0(E_hat, tau, N, M, b0);

    // B
    E_hat = Y - one_N * b0.transpose() - mphi * mtheta.transpose();
    mv_lm_uninf_gibbs_B(E_hat, X, tau, M, P, B);

    // tau
    E_hat = Y - one_N * b0.transpose() - X * B.transpose() -
      mphi * mtheta.transpose();
    mv_lm_uninf_gibbs_tau(E_hat, N, M, a_tau, b_tau, tau);

    // mgamma
    mv_lm_uninf_gibbs_mgamma(mtheta, lambda, M, K, nu, mgamma);

    // xi and lambda
    mv_lm_uninf_gibbs_xi(mtheta, mgamma, lambda, M, K, a1, a2, xi);
    lambda = cum_prod(xi);

    //--------------------------------------------------
    // store values
    //--------------------------------------------------
    // B, mphi, mtheta, mgamma, xi, and tau
    Eigen::Map<VectorXd> B_vec(B.data(), B.size());
    Eigen::Map<VectorXd> theta_vec(mtheta.data(), mtheta.size());

    b0_mat.row(i) = b0;
    B_mat.row(i) = B_vec;
    mtheta_mat.row(i) = theta_vec;
    tau_mat.row(i) = tau;

    // log likelihood
    D_temp.diagonal() = tau;
    E_hat = Y - one_N * b0.transpose() - X * B.transpose() -
      mphi * mtheta.transpose();
    loglik_vec(i) = 1.0/2.0 * tau.array().log().sum() -
      1.0 / 2.0 * (D_temp * E_hat.transpose() * E_hat).trace();
  }
  Rcpp::List retl;
  retl["b0_mat"] = b0_mat;
  retl["B_mat"] = B_mat;
  retl["mtheta_mat"] = mtheta_mat;
  retl["tau_mat"] = tau_mat;
  retl["loglik_vec"] = loglik_vec;
  return retl;
}


