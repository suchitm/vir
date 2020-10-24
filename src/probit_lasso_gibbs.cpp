#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include "typedefs.h"
#include "include/helpers.hpp"
#include "include/probit_helpers.hpp"

// *****************************************************************************
// individual samplers
// *****************************************************************************
Eigen::VectorXd probit_lasso_gibbs_gamma(
  double& lambda2, Eigen::VectorXd& b, int& P, Eigen::VectorXd& gamma
){
  for(int p = 0; p < P; p++)
  {
    double mu = std::sqrt(lambda2 / (b(p) * b(p)));
    gamma(p) = rinvgauss_cpp(1, mu, lambda2)(0);
  }
  return(gamma);
}

double probit_lasso_gibbs_lambda2(
  Eigen::VectorXd& gamma, double& a_lambda2, double& b_lambda2, int& P,
  double& lambda2
){
  double shape = P + a_lambda2;
  double rate = 1/2.0 * gamma.array().inverse().sum() + b_lambda2;
  lambda2 = Rcpp::rgamma(1, shape, 1/rate)(0);
  return(lambda2);
}

//**********************************************************************//
// Main function
//**********************************************************************//
//' Univariate probit linear regression with a LASSO prior using a
//' Gibbs sampler.
//' @param y Vector or responses (N by 1)
//' @param X Matrix of predictors (N by P)
//' @param n_iter Number of iterations to run the algorithm for
//' @param verbose True or False. Do you want to print messages along the way?
//' @param a_lambda2 Prior shape parameter for the coefficient precision
//'   (shrinkage) term.
//' @param b_lambda2 Prior rate parameter for the coefficient precision
//'   (shrinkage) term.
//' @export
// [[Rcpp::export]]
Rcpp::List probit_lasso_gibbs(
  Eigen::VectorXi y, Eigen::MatrixXd X, bool verbose = true,
  int n_iter = 10000, double a_lambda2 = 0.1, double b_lambda2 = 0.1
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
  Eigen::MatrixXd z_mat = Eigen::MatrixXd::Constant(n_iter, N, 1);
  Eigen::VectorXd b0_vec = Eigen::VectorXd::Constant(n_iter, 1);
  Eigen::MatrixXd b_mat = Eigen::MatrixXd::Constant(n_iter, P, 1);
  Eigen::MatrixXd gamma_mat = Eigen::MatrixXd::Constant(n_iter, P, 1);
  Eigen::VectorXd lambda2_vec = Eigen::VectorXd::Constant(n_iter, 1);
  Eigen::VectorXd log_lik_vec = Eigen::VectorXd::Constant(n_iter, 1);

  // starting values
  double b0 = 0.0;
  double lambda2 = 1.0;
  Eigen::VectorXd z = Eigen::VectorXd::Constant(N, 0.0);
  Eigen::VectorXd b = Eigen::VectorXd::Constant(P, 0.0);
  Eigen::VectorXd gamma = Eigen::VectorXd::Constant(P, 1.0);
  Eigen::VectorXd eta = ones * b0 + X * b;
  Eigen::VectorXd ehat(N);
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
    prior_mat = gamma.asDiagonal();
    probit_gibbs_b(X, ehat, prior_mat, P, b);

    // global shrinkage parameter
    probit_lasso_gibbs_lambda2(gamma, a_lambda2, b_lambda2, P, lambda2);

    // double exponential latent variables
    probit_lasso_gibbs_gamma(lambda2, b, P, gamma);

    // store results
    z_mat.row(i) = z;
    b0_vec(i) = b0 - (b.array() * s_x.array()).sum();
    b_mat.row(i) = b.transpose().array() / vsigma_x.array();
    gamma_mat.row(i) = gamma;
    lambda2_vec(i) = lambda2;
    log_lik_vec(i) = probit_log_lik(y, X, b0, b, N);
  }
  Rcpp::List ret;
  ret["z_mat"] = z_mat;
  ret["b0_vec"] = b0_vec;
  ret["b_mat"] = b_mat;
  ret["lambda2_vec"] = lambda2_vec;
  ret["gamma_mat"] = gamma_mat;
  ret["log_lik_vec"] = log_lik_vec;

  return(ret);
}
