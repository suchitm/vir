#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include "typedefs.h"
#include "include/helpers.hpp"

using namespace Eigen;
using namespace Rcpp;
using namespace std;

// **********************************************************************
// individual updaters
// **********************************************************************
Rcpp::List lm_lasso_vi_b0(
  Eigen::VectorXd& y_n, Eigen::MatrixXd& X_n, Rcpp::List& param_b,
  Rcpp::List& param_tau, int& N, int& S, Rcpp::List& param_b0
){
  double mu_tau = param_tau["mu"];
  Eigen::VectorXd mu_b = param_b["mu"];

  double delta1_t = N * 1.0 / S * mu_tau * (y_n - X_n * mu_b).sum();
  double delta2_t = -1.0 / 2.0 * (N * mu_tau + 0.000001);

  param_b0["delta1_t"] = delta1_t;
  param_b0["delta2_t"] = delta2_t;
  return(param_b0);
}

Rcpp::List lm_lasso_vi_b(
  Eigen::VectorXd& y_n, Eigen::MatrixXd& X_n, Rcpp::List& param_b0,
  Rcpp::List& param_tau, Rcpp::List& param_gamma, int& N, int& P, int& S,
  int& type, Rcpp::List& param_b, bool svi = false
){
  Eigen::VectorXd delta1_t = param_b["delta1_t"];
  Eigen::MatrixXd delta2_t0(P, P);
  Eigen::VectorXd delta2_t1(P);
  double mu_b0 = param_b0["mu"];
  double mu_tau = param_tau["mu"];
  Eigen::VectorXd mu_gamma = param_gamma["mu"];
  Eigen::MatrixXd gamma_mat = mu_gamma.asDiagonal();
  Eigen::VectorXd one_S = Eigen::VectorXd::Constant(S, 1.0);
  Eigen::VectorXd ehat_np(S);
  Eigen::VectorXd mu_b = param_b["mu"];

  if(type == 0)
  {
    Eigen::MatrixXd delta2_t = param_b["delta2_t"];
    delta1_t = mu_tau * N * 1.0 / S * X_n.transpose() * (y_n - one_S * mu_b0);
    delta2_t0 = -mu_tau / 2.0 / S * (N * X_n.transpose() * X_n + S * gamma_mat);
    param_b["delta2_t"] = delta2_t0;
  }
  else
  {
    for(int p = 0; p < P; p++)
    {
      ehat_np = y_n - one_S * mu_b0 - X_n.leftCols(p) * mu_b.head(p) -
        X_n.rightCols(P - p - 1) * mu_b.tail(P - p - 1);
      delta1_t(p) = N * mu_tau / S * X_n.col(p).transpose() * ehat_np;
      delta2_t1(p) = -mu_tau / 2.0 / S * (
        N * X_n.col(p).array().square().sum() + S * mu_gamma(p)
      );
      if(!svi) {
        mu_b(p) = -1.0 / 2.0 * delta1_t(p) / delta2_t1(p);
      }
    }
    param_b["delta2_t"] = delta2_t1;
  }
  param_b["delta1_t"] = delta1_t;
  return(param_b);
}

Rcpp::List lm_lasso_vi_tau(
  Eigen::VectorXd& y_n, Eigen::MatrixXd& X_n, Rcpp::List& param_b0,
  Rcpp::List& param_b, Rcpp::List& param_gamma, int& N, int& P, int& S,
  double& a_tau, double& b_tau, Rcpp::List& param_tau
){
  double mu_b0 = param_b0["mu"];
  double sigma2_b0 = param_b0["sigma2"];
  Eigen::VectorXd mu_b = param_b["mu"];
  Eigen::MatrixXd msigma_b = param_b["msigma"];
  Eigen::VectorXd mu_gamma = param_gamma["mu"];
  double delta1_t = param_tau["delta1_t"];
  double delta2_t = param_tau["delta2_t"];

  delta1_t = (N + P) / 2.0 + a_tau - 1.0;
  delta2_t = -b_tau - 1.0 / 2.0 * (
    N * 1.0 / S * (
      (y_n - Eigen::VectorXd::Ones(S, 1) * mu_b0 - X_n * mu_b).squaredNorm() +
      S * sigma2_b0 +
      (X_n.transpose() * X_n * msigma_b).trace()
    ) + (
      mu_gamma.array() * (mu_b.array().square() + msigma_b.diagonal().array())
    ).sum()
  );

  param_tau["delta1_t"] = delta1_t;
  param_tau["delta2_t"] = delta2_t;
  return(param_tau);
}

Rcpp::List lm_lasso_vi_lambda2(
  Rcpp::List& param_b, Rcpp::List& param_gamma, int& P, double& a_lambda2,
  double& b_lambda2, Rcpp::List& param_lambda2
){
  Eigen::VectorXd mu_inv_gamma = param_gamma["mu_inv"];
  double delta1_t = P + a_lambda2 - 1;
  double delta2_t = -b_lambda2 - 1.0 / 2.0 * mu_inv_gamma.sum();

  param_lambda2["delta1_t"] = delta1_t;
  param_lambda2["delta2_t"] = delta2_t;

  return(param_lambda2);
}

Rcpp::List lm_lasso_vi_gamma(
  Rcpp::List& param_b, Rcpp::List& param_tau, Rcpp::List& param_lambda2,
  int& P, Rcpp::List& param_gamma
){
  Eigen::VectorXd mu_b = param_b["mu"];
  Eigen::VectorXd vsigma2_b = param_b["vsigma2"];
  double mu_tau = param_tau["mu"];
  double mu_lambda2 = param_lambda2["mu"];
  Eigen::VectorXd delta1_t = param_gamma["delta1_t"];
  Eigen::VectorXd delta2_t = param_gamma["delta2_t"];

  delta1_t = -mu_tau / 2.0 * (
    mu_b.array().square() + vsigma2_b.array()
  );
  delta2_t = -mu_lambda2 / 2.0 * Eigen::VectorXd::Constant(P, 1.0);

  param_gamma["delta1_t"] = delta1_t;
  param_gamma["delta2_t"] = delta2_t;
  return(param_gamma);
}

double lm_lasso_vi_elbo(
  Eigen::MatrixXd& X, Eigen::VectorXd& y, Rcpp::List& param_b0,
  Rcpp::List& param_b, Rcpp::List& param_tau, Rcpp::List& param_lambda2,
  Rcpp::List& param_gamma, double& a_lambda2, double& b_lambda2,
  double& a_tau, double& b_tau, int& N, int& P
){
  double mu_b0 = param_b0["mu"];
  double sigma2_b0 = param_b0["sigma2"];
  Eigen::VectorXd mu_b = param_b["mu"];
  Eigen::VectorXd vsigma2_b = param_b["vsigma2"];
  Eigen::MatrixXd msigma_b = param_b["msigma"];
  double logdet_msigma_b = param_b["logdet_msigma"];
  double mu_tau = param_tau["mu"];
  double astar_tau = param_tau["shape"];
  double bstar_tau = param_tau["rate"];
  double mu_lambda2 = param_lambda2["mu"];
  double astar_lambda2 = param_lambda2["shape"];
  double bstar_lambda2 = param_lambda2["rate"];
  Eigen::VectorXd mu_gamma = param_gamma["mu"];
  Eigen::VectorXd mu_gamma_inv = param_gamma["mu_inv"];


  double ll = -N/2.0 * std::log(2.0 * M_PI) +
    N/2.0 * (Rf_digamma(astar_tau) - std::log(bstar_tau)) -
    mu_tau/2.0 * (
      (y - mu_b0 * Eigen::VectorXd::Ones(P) - X * mu_b).squaredNorm() +
      N * sigma2_b0 +
      (X.transpose() * X * msigma_b).trace()
    );

  double lp_b0 = -1.0/2.0 * std::log(2.0 * M_PI) -
    3 * std::log(10.0) -
    0.000001/2.0 * (mu_b0 * mu_b0 + sigma2_b0);

  double lp_b = -P/2.0 * std::log(2.0 * M_PI) +
    P/2.0 * (Rf_digamma(astar_tau) - std::log(bstar_tau)) -
    mu_tau/2.0 * (
      mu_b.transpose() * mu_gamma.asDiagonal() * mu_b +
      (msigma_b * mu_gamma.asDiagonal()).trace()
    );

  double lp_tau = a_tau * std::log(b_tau) - Rf_lgammafn(a_tau) +
    (a_tau - 1) * (Rf_digamma(astar_tau) - std::log(bstar_tau)) -
    b_tau * mu_tau;

  double lp_lambda2 = a_lambda2 * std::log(b_lambda2) - Rf_lgammafn(a_lambda2) +
    (a_lambda2 - 1) * (Rf_digamma(astar_lambda2) - std::log(bstar_lambda2)) -
    b_lambda2 * mu_lambda2;

  double lp_gamma = (-1.0 * mu_lambda2 / 2.0 * mu_gamma_inv +
    Eigen::VectorXd::Ones(P) * (
      Rf_digamma(astar_lambda2) - std::log(bstar_lambda2) - std::log(2.0)
    )).sum();

  double lq_b0 = -1.0/2.0 * std::log(sigma2_b0) -
    1.0/2.0 * (std::log(2 * M_PI) + 1.0);

  double lq_b = -1.0/2.0 * logdet_msigma_b -
    P/2.0 * (std::log(2 * M_PI) + 1.0);

  double lq_tau = -Rf_lgammafn(astar_tau) + std::log(bstar_tau) +
    (astar_tau - 1) * Rf_digamma(astar_tau) - astar_tau;

  double lq_lambda2 = -Rf_lgammafn(astar_lambda2) + std::log(bstar_lambda2) +
    (astar_lambda2 - 1) * Rf_digamma(astar_lambda2) - astar_lambda2;

  double lq_gamma = 0.0;
  for(int p = 0; p < P; p++)
  {
    lq_gamma += 1.0/2.0 * (
      std::log(mu_lambda2) - std::log(2.0 * M_PI) +
      mu_lambda2 / mu_gamma(p) -
      mu_lambda2 * mu_gamma_inv(p)
    );
  }

  double elbo = ll + lp_b0 + lp_b + lp_tau + lp_lambda2 + lp_gamma -
    lq_b0 - lq_b - lq_tau - lq_lambda2 - lq_gamma;

  return(elbo);
}

// **********************************************************************
// CAVI
// **********************************************************************
//' Univariate normal linear regression with a LASSO (double-exponential) prior
//' using the CAVI algorithm.
//' @param y Vector or responses (N by 1)
//' @param X Matrix of predictors (N by P)
//' @param n_iter Max number of iterations to run the algorithm for (default =
//'   1000). A convergence warning is issues if the algorithm runs for the max
//'   number of iterations.
//' @param verbose True or False. Do you want to print messages along the way?
//' @param a_tau Prior shape parameter for the likelihood precision.
//' @param b_tau Prior rate parameter for the likelihood precision.
//' @param a_lambda2 Prior shape parameter for the coefficient precision
//'   (shrinkage) term.
//' @param b_lambda2 Prior rate parameter for the coefficient precision
//'   (shrinkage) term.
//' @param rel_tol Relative tolerance used for convergence. Convergence is
//'   assesed using the evidence lower bound (ELBO) changes relative to five
//'   iterations prior.
//' @param type Correlation structure of the regression coefficients. Use 0 for
//'   full correlation and 1 for independece assumption.
//' @export
// [[Rcpp::export]]
Rcpp::List lm_lasso_cavi(
  Eigen::VectorXd y, Eigen::MatrixXd X, bool verbose = true,
  int n_iter = 1000, double a_tau = 0.1, double b_tau = 0.1,
  double a_lambda2 = 0.1, double b_lambda2 = 0.1, double rel_tol = 0.0001,
  int type = 0
){

  // problem info
  int N = X.rows();
  int P = X.cols();
  int S = N;

  // scale X
  Eigen::RowVectorXd vmu_x = X.colwise().mean();
  Eigen::RowVectorXd vsigma_x =
    (X.rowwise() - vmu_x).colwise().squaredNorm() / (X.rows() - 1);
  vsigma_x = vsigma_x.array().sqrt();
  Eigen::VectorXd vsigma_x_inv = vsigma_x.transpose().cwiseInverse();
  Eigen::VectorXd s_x = vmu_x.array() / vsigma_x.array();
  X = (X.rowwise() - vmu_x).array().rowwise() / vsigma_x.array();

  // initializing starting values for parameters
  Rcpp::List param_b0 = vi_init_normal();
  Rcpp::List param_b = vi_init_mv_normal(P, type);
  Rcpp::List param_tau = vi_init_gamma(1);
  Rcpp::List param_lambda2 = vi_init_gamma(1);
  Rcpp::List param_gamma = vi_init_inv_gauss(P);

  // vectors for storage
  Eigen::VectorXd elbo = Eigen::VectorXd::Constant(n_iter, 1);
  double rhot = 1.0;

  // main loop of algorithm
  int iters = 0;
  for(int i = 0; i < n_iter; i++)
  {

    if(verbose && (i % 100 == 0)) {
      Rcpp::Rcout << "Done with Iteration " << i << " of " << n_iter << "\n";
    }

    // b0
    lm_lasso_vi_b0(y, X, param_b, param_tau, N, S, param_b0);
    vi_update_normal(param_b0, rhot);
    natural_to_canonical(param_b0, "normal");

    // b
    lm_lasso_vi_b(
      y, X, param_b0, param_tau, param_gamma, N, P, S, type, param_b, false
    );
    vi_update_mv_normal(param_b, type, rhot);
    if(type == 0) {
      natural_to_canonical(param_b, "mv_normal");
    } else {
      natural_to_canonical(param_b, "mv_normal_ind");
    }

    // tau
    lm_lasso_vi_tau(
      y, X, param_b0, param_b, param_gamma, N, P, S, a_tau, b_tau, param_tau
    );
    vi_update_gamma(param_tau, rhot);
    natural_to_canonical(param_tau, "univ_gamma");

    // lambda2
    lm_lasso_vi_lambda2(
      param_b, param_gamma, P, a_lambda2, b_lambda2, param_lambda2
    );
    vi_update_gamma(param_lambda2, rhot);
    natural_to_canonical(param_lambda2, "univ_gamma");

    // gamma
    lm_lasso_vi_gamma(param_b, param_tau, param_lambda2, P, param_gamma);
    vi_update_inv_gauss(param_gamma, rhot);
    natural_to_canonical(param_gamma, "inv_gauss");

    // calculate elbo
    elbo(i) = lm_lasso_vi_elbo(
      X, y, param_b0, param_b, param_tau, param_lambda2, param_gamma,
      a_lambda2, b_lambda2, a_tau, b_tau, N, P
    );

    iters = iters + 1;
    if(i > 4)
    {
      // check if lower bound decreases
      if(elbo(i) < elbo(i-1))
        std::cout << "LOWER BOUND DECREASES" << "\n";
      if((1.0 - elbo(i) / elbo(i-5)) < rel_tol) { break; }
      if(i == (n_iter - 1))
        std::cout << "VB DID NOT CONVERGE" << "\n";
    }
    Rcpp::checkUserInterrupt();
  }

  // values to rescale
  double mu_b0 = param_b0["mu"];
  double sigma2_b0 = param_b0["sigma2"];
  Eigen::VectorXd mu_b = param_b["mu"];
  Eigen::MatrixXd msigma_b = param_b["msigma"];
  double astar_lambda2 = param_lambda2["shape"];
  double bstar_lambda2 = param_lambda2["rate"];
  double astar_tau = param_tau["shape"];
  double bstar_tau = param_tau["rate"];
  Eigen::VectorXd mu_gamma = param_gamma["mu"];
  Eigen::VectorXd lambda_gamma = param_gamma["lambda"];

  // rescaled values - need mu_b0, sigma2_b0, msigma_b
  mu_b0 = mu_b0 - (mu_b.array() * s_x.array()).sum();
  sigma2_b0 = sigma2_b0 + s_x.transpose() * msigma_b * s_x;
  mu_b = mu_b.array() * vsigma_x_inv.array();
  msigma_b = vsigma_x_inv.asDiagonal() * msigma_b * vsigma_x_inv.asDiagonal();

  List b0;
  b0["dist"] = "univariate normal";
  b0["mu"] = mu_b0;
  b0["var"] = sigma2_b0;

  List b;
  b["dist"] = "multivariate normal";
  b["mu"] = mu_b;
  b["sigma_mat"] = msigma_b;

  List tau;
  tau["dist"] = "gamma";
  tau["shape"] = astar_tau;
  tau["rate"] = bstar_tau;

  List lambda2;
  lambda2["dist"] = "gamma";
  lambda2["shape"] = astar_lambda2;
  lambda2["rate"] = bstar_lambda2;

  List gamma;
  gamma["dist"] = "inverse gaussian";
  gamma["mu"] = mu_gamma;
  gamma["lambda"] = lambda_gamma;

  List ret;
  ret["b0"] = b0;
  ret["b"] = b;
  ret["tau"] = tau;
  ret["lambda2"] = lambda2;
  ret["gamma"] = gamma;
  ret["elbo"] = elbo.topRows(iters);
  return(ret);

}

// **********************************************************************
// SVI
// **********************************************************************
//' Univariate normal linear regression with a ridge (normal) prior using the
//' SVI algorithm.
//' @param y Vector or responses (N by 1)
//' @param X Matrix of predictors (N by P)
//' @param n_iter Number of iterations to run the algorithm for (default =
//'   5000).
//' @param verbose True or False. Do you want to print messages along the way?
//' @param a_tau Prior shape parameter for the likelihood precision.
//' @param b_tau Prior rate parameter for the likelihood precision.
//' @param a_lambda2 Prior shape parameter for the coefficient precision
//'   (shrinkage) term.
//' @param b_lambda2 Prior rate parameter for the coefficient precision
//'   (shrinkage) term.
//' @param type Correlation structure of the regression coefficients. Use 0 for
//'   full correlation and 1 for independece assumption.
//' @param batch_size Size of the subsamples used to update the parameters.
//' @param cost_rhot Used to set a constant step size in the gradient descent
//'   algorithm. If this parameter is greater than zero, it overrides the step
//'   size iterations calculated using kappa and omega.
//' @param omega Delay for the stepsize (\eqn{\omega}) for the gradient step.
//'   Interacts with \eqn{\kappa} via the formula \eqn{\rho_{t} = (t +
//'   \omega)^{-\kappa}}. This parameter has to be greater than or equal to zero.
//' @param kappa Forgetting rate for the step size iterations; \eqn{\kappa \in
//'   (0.5, 1)}
//' @export
// [[Rcpp::export]]
Rcpp::List lm_lasso_svi(
  Eigen::VectorXd y, Eigen::MatrixXd X, bool verbose = true,
  int n_iter = 1000, double a_tau = 0.1, double b_tau = 0.1,
  double a_lambda2 = 0.1, double b_lambda2 = 0.1, int type = 0,
  int batch_size = 10, double const_rhot = 0.01, double omega = 15.0,
  double kappa = 0.6
){

  // problem info
  int N = X.rows();
  int P = X.cols();
  int S = batch_size;

  // scale X
  Eigen::RowVectorXd vmu_x = X.colwise().mean();
  Eigen::RowVectorXd vsigma_x =
    (X.rowwise() - vmu_x).colwise().squaredNorm() / (X.rows() - 1);
  vsigma_x = vsigma_x.array().sqrt();
  Eigen::VectorXd vsigma_x_inv = vsigma_x.transpose().cwiseInverse();
  Eigen::VectorXd s_x = vmu_x.array() / vsigma_x.array();
  X = (X.rowwise() - vmu_x).array().rowwise() / vsigma_x.array();

  // initializing starting values for parameters
  Rcpp::List param_b0 = vi_init_normal();
  Rcpp::List param_b = vi_init_mv_normal(P, type);
  Rcpp::List param_tau = vi_init_gamma(1);
  Rcpp::List param_lambda2 = vi_init_gamma(1);
  Rcpp::List param_gamma = vi_init_inv_gauss(P);

  // svi specific storage
  Eigen::VectorXd y_s(S);
  Eigen::MatrixXd X_s(S, P);
  Rcpp::IntegerVector the_sample = seq(0, S - 1);
  Rcpp::IntegerVector seq_samp = seq(0, N - 1);
  double rhot;

  // main loop of algorithm
  for(int i = 0; i < n_iter; i++)
  {

    // sample data
    the_sample = sample(seq_samp, S, false);
    get_subsample_lm(y, X, y_s, X_s, S, N, the_sample);

    if(const_rhot <= 0)
      rhot = std::exp(-kappa * std::log(i + 1.0 + omega));
    else
      rhot = const_rhot;

    // calculate updates
    // b0
    lm_lasso_vi_b0(y_s, X_s, param_b, param_tau, N, S, param_b0);
    // b
    lm_lasso_vi_b(
      y_s, X_s, param_b0, param_tau, param_gamma, N, P, S, type, param_b, false
    );
    // tau
    lm_lasso_vi_tau(
      y_s, X_s, param_b0, param_b, param_gamma, N, P, S, a_tau, b_tau,
      param_tau
    );
    // lambda2
    lm_lasso_vi_lambda2(
      param_b, param_gamma, P, a_lambda2, b_lambda2, param_lambda2
    );
    // gamma
    lm_lasso_vi_gamma(param_b, param_tau, param_lambda2, P, param_gamma);

    // gradient step
    vi_update_normal(param_b0, rhot);
    vi_update_mv_normal(param_b, type, rhot);
    vi_update_gamma(param_tau, rhot);
    vi_update_gamma(param_lambda2, rhot);
    vi_update_inv_gauss(param_gamma, rhot);

    // calculate means and variances for next iteration
    natural_to_canonical(param_b0, "normal");
    if(type == 0) {
      natural_to_canonical(param_b, "mv_normal");
    } else {
      natural_to_canonical(param_b, "mv_normal_ind");
    }
    natural_to_canonical(param_tau, "univ_gamma");
    natural_to_canonical(param_lambda2, "univ_gamma");
    natural_to_canonical(param_gamma, "inv_gauss");

    if(verbose && (i % 100 == 0)) {
      Rcpp::Rcout << "Done with Iteration " << i << " of " << n_iter << "\n";
      Rcpp::Rcout << "rhot: " << rhot << std::endl;
    }
    Rcpp::checkUserInterrupt();

    //// calculate elbo
    //elbo(i) = lm_lasso_vi_elbo(
    //  X, y, param_b0, param_b, param_tau, param_lambda2, param_gamma,
    //  a_lambda2, b_lambda2, a_tau, b_tau, N, P
    //);

    //iters = iters + 1;
    //if(i > 4)
    //{
    //  // check if lower bound decreases
    //  if(elbo(i) < elbo(i-1))
    //    std::cout << "LOWER BOUND DECREASES" << "\n";
    //  if(std::abs(elbo(i) - elbo(i-5)) < tol) { break; }
    //  if(i == (n_iter - 1))
    //    std::cout << "VB DID NOT CONVERGE" << "\n";
    //}
  }

  // values to rescale
  double mu_b0 = param_b0["mu"];
  double sigma2_b0 = param_b0["sigma2"];
  Eigen::VectorXd mu_b = param_b["mu"];
  Eigen::MatrixXd msigma_b = param_b["msigma"];
  double astar_lambda2 = param_lambda2["shape"];
  double bstar_lambda2 = param_lambda2["rate"];
  double astar_tau = param_tau["shape"];
  double bstar_tau = param_tau["rate"];
  Eigen::VectorXd mu_gamma = param_gamma["mu"];
  Eigen::VectorXd lambda_gamma = param_gamma["lambda"];

  // rescaled values - need mu_b0, sigma2_b0, msigma_b
  mu_b0 = mu_b0 - (mu_b.array() * s_x.array()).sum();
  sigma2_b0 = sigma2_b0 + s_x.transpose() * msigma_b * s_x;
  mu_b = mu_b.array() * vsigma_x_inv.array();
  msigma_b = vsigma_x_inv.asDiagonal() * msigma_b * vsigma_x_inv.asDiagonal();

  List b0;
  b0["dist"] = "univariate normal";
  b0["mu"] = mu_b0;
  b0["var"] = sigma2_b0;

  List b;
  b["dist"] = "multivariate normal";
  b["mu"] = mu_b;
  b["sigma_mat"] = msigma_b;

  List tau;
  tau["dist"] = "gamma";
  tau["shape"] = astar_tau;
  tau["rate"] = bstar_tau;

  List lambda2;
  lambda2["dist"] = "gamma";
  lambda2["shape"] = astar_lambda2;
  lambda2["rate"] = bstar_lambda2;

  List gamma;
  gamma["dist"] = "inverse gaussian";
  gamma["mu"] = mu_gamma;
  gamma["lambda"] = lambda_gamma;

  List ret;
  ret["b0"] = b0;
  ret["b"] = b;
  ret["tau"] = tau;
  ret["lambda2"] = lambda2;
  ret["gamma"] = gamma;
  return(ret);
}



