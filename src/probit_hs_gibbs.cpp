#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include "typedefs.h"
#include "include/helpers.hpp"
#include "include/probit_helpers.hpp"

// *****************************************************************************
// individual samplers
// *****************************************************************************
double probit_hs_gibbs_lambda(
  Eigen::VectorXd& b, Eigen::VectorXd& gamma, double& xi,
  int& P, double& lambda
){
  double shape = (P + 1) / 2.0;
  double rate = xi +
    1.0 / 2.0 * (gamma.array() * b.array().square()).sum();
  lambda = Rcpp::rgamma(1, shape, 1/rate)(0);
  return(lambda);
}

Eigen::VectorXd probit_hs_gibbs_gamma(
  double& lambda, Eigen::VectorXd& nu, Eigen::VectorXd& b,
  int& P, Eigen::VectorXd& gamma
){
  for(int p = 0; p < P; p++)
  {
    double shape = 1.0;
    double rate = nu(p) + b(p) * b(p) * lambda / 2.0;
    gamma(p) = Rcpp::rgamma(1, shape, 1/rate)(0);
  }
  return(gamma);
}

double probit_hs_gibbs_xi(double& lambda, double& xi)
{
  xi = Rcpp::rgamma(1, 1.0, 1.0 / (1.0 + lambda))(0);
  return(xi);
}

Eigen::VectorXd probit_hs_gibbs_nu(
  Eigen::VectorXd& gamma, int& P, Eigen::VectorXd& nu
){
  for(int p = 0; p < P; p++)
  {
    nu(p) = Rcpp::rgamma(1, 1.0, 1.0 / (1.0 + gamma(p)))(0);
  }
  return(nu);
}

// *****************************************************************************
// full gibbs sampler
// *****************************************************************************
//' Univariate probit linear regression with a Horseshoe prior using a
//' Gibbs sampler.
//' @param y Vector or responses (N by 1)
//' @param X Matrix of predictors (N by P)
//' @param n_iter Number of iterations to run the algorithm for
//' @param verbose True of False. Do you want to print messages along the way?
//'   (shrinkage) term.
//' @export
// [[Rcpp::export]]
Rcpp::List probit_hs_gibbs(
  Eigen::VectorXi y, Eigen::MatrixXd X, bool verbose = true, int n_iter = 10000
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
  Eigen::VectorXd b0_vec = Eigen::VectorXd::Constant(n_iter, 1.0);
  Eigen::MatrixXd b_mat = Eigen::MatrixXd::Constant(n_iter, P, 1.0);
  Eigen::VectorXd lambda_vec = Eigen::VectorXd::Constant(n_iter, 1.0);
  Eigen::VectorXd xi_vec = Eigen::VectorXd::Constant(n_iter, 1.0);
  Eigen::MatrixXd gamma_mat = Eigen::MatrixXd::Constant(n_iter, P, 1.0);
  Eigen::MatrixXd nu_mat = Eigen::MatrixXd::Constant(n_iter, P, 1.0);
  Eigen::VectorXd log_lik_vec = Eigen::VectorXd::Constant(n_iter, 1.0);

  // starting values
  Eigen::VectorXd z = Eigen::VectorXd::Constant(N, 0);
  double b0 = 0.0;
  Eigen::VectorXd b = Eigen::VectorXd::Constant(P, 0);
  double lambda = 1.0;
  double xi = 1.0;
  Eigen::VectorXd gamma = Eigen::VectorXd::Constant(P, 1.0);
  Eigen::VectorXd nu = Eigen::VectorXd::Constant(P, 1.0);
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
    prior_mat = lambda * gamma.asDiagonal();
    probit_gibbs_b(X, ehat, prior_mat, P, b);

    // global shrinkage parameter
    probit_hs_gibbs_lambda(b, gamma, xi, P, lambda);

    // double exponential latent variables
    probit_hs_gibbs_gamma(lambda, nu, b, P, gamma);

    // hyperpriors
    probit_hs_gibbs_xi(lambda, xi);
    probit_hs_gibbs_nu(gamma, P, nu);

    // store results
    z_mat.row(i) = z;
    b0_vec(i) = b0 - (b.array() * s_x.array()).sum();
    b_mat.row(i) = b.transpose().array() / vsigma_x.array();
    lambda_vec(i) = lambda;
    xi_vec(i) = xi;
    gamma_mat.row(i) = gamma;
    nu_mat.row(i) = nu;
    log_lik_vec(i) = probit_log_lik(y, X, b0, b, N);
  }

  Rcpp::List ret;
  ret["z_mat"] = z_mat;
  ret["b0_vec"] = b0_vec;
  ret["b_mat"] = b_mat;
  ret["lambda_vec"] = lambda_vec;
  ret["xi_vec"] = xi_vec;
  ret["gamma_mat"] = gamma_mat;
  ret["nu_mat"] = nu_mat;
  ret["log_lik_vec"] = log_lik_vec;

  return(ret);
}
