#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include "typedefs.h"
#include "include/helpers.hpp"
#include "include/probit_helpers.hpp"

// *****************************************************************************
// individual samplers
// *****************************************************************************
double probit_ridge_gibbs_lambda(
  Eigen::VectorXd& b, double& a_lambda, double& b_lambda, int& P,
  double& lambda
){
  double shape = P/2.0 + a_lambda;
  double rate = 1/2.0 * b.squaredNorm() + b_lambda;
  lambda = Rcpp::rgamma(1, shape, 1/rate)(0);
  return(lambda);
}

// *****************************************************************************
// full gibbs sampler
// *****************************************************************************
//' Univariate probit linear regression with a ridge (normal) prior using a
//' Gibbs sampler.
//' @param y Vector or responses (N by 1)
//' @param X Matrix of predictors (N by P)
//' @param n_iter Number of iterations to run the algorithm for
//' @param verbose True or False. Do you want to print messages along the way?
//' @param a_lambda Prior shape parameter for the coefficient precision
//'   (shrinkage) term.
//' @param b_lambda Prior rate parameter for the coefficient precision
//'   (shrinkage) term.
//' @export
// [[Rcpp::export]]
Rcpp::List probit_ridge_gibbs(
  Eigen::VectorXi y, Eigen::MatrixXd X, bool verbose = true,
  int n_iter = 10000, double a_lambda = 0.01, double b_lambda = 0.01
){
  int N = X.rows();
  int P = X.cols();
  Eigen::VectorXd ones = Eigen::VectorXd::Constant(N, 1);

  // scale X
  Eigen::RowVectorXd vmu_x = X.colwise().mean();
  Eigen::RowVectorXd vsigma_x =
      (X.rowwise() - vmu_x).colwise().squaredNorm() / (X.rows() - 1);
  vsigma_x = vsigma_x.array().sqrt();
  Eigen::VectorXd s_x = vmu_x.array() / vsigma_x.array();
  X = (X.rowwise() - vmu_x).array().rowwise() / vsigma_x.array();

  // initializing matricies to store results
  Eigen::VectorXd b0_vec = Eigen::VectorXd::Constant(n_iter, 1);
  Eigen::MatrixXd b_mat = Eigen::MatrixXd::Constant(n_iter, P, 1);
  Eigen::MatrixXd z_mat = Eigen::MatrixXd::Constant(n_iter, N, 1);
  Eigen::VectorXd lambda_vec = Eigen::VectorXd::Constant(n_iter, 1);
  Eigen::VectorXd log_lik_vec = Eigen::VectorXd::Constant(n_iter, 1);

  // starting values
  double b0 = 0.0;
  double lambda = 1.0;
  Eigen::VectorXd z = Eigen::VectorXd::Constant(N, 0);
  Eigen::VectorXd b = Eigen::VectorXd::Constant(P, 0);
  Eigen::VectorXd eta = ones * b0 + X * b;
  Eigen::VectorXd ehat(N);
  Eigen::MatrixXd id_mat = Eigen::MatrixXd::Identity(P, P);
  Eigen::MatrixXd prior_mat(P, P);

  // main loop of sampler
  for(int i = 0; i < n_iter; i++)
  {
    Rcpp::checkUserInterrupt();
    if(verbose && (i % 1000 == 0)) {
      Rcpp::Rcout << "Done with Iteration " << i << " of " << n_iter << "\r";
    }

    // update z
    eta = ones * b0 + X * b;
    probit_gibbs_z(y, eta, N, z);

    // sample the mean
    ehat = z - X * b;
    probit_gibbs_b0(ehat, N, b0);

    // update the coefs - b
    ehat = z - ones * b0;
    prior_mat = lambda * id_mat;
    probit_gibbs_b(X, ehat, prior_mat, P, b);

    // update lambda
    probit_ridge_gibbs_lambda(b, a_lambda, b_lambda, P, lambda);

    // store results
    z_mat.row(i) = z;
    b0_vec(i) = b0 - (b.array() * s_x.array()).sum();
    b_mat.row(i) = b.transpose().array() / vsigma_x.array();
    lambda_vec(i) = lambda;
    log_lik_vec(i) = probit_log_lik(y, X, b0, b, N);
  }
  Rcpp::List ret;
  ret["z_mat"] = z_mat;
  ret["b0_vec"] = b0_vec;
  ret["b_mat"] = b_mat;
  ret["lambda_vec"] = lambda_vec;
  ret["log_lik_vec"] = log_lik_vec;

  return(ret);
}
