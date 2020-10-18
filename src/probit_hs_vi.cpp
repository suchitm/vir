#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include "typedefs.h"
#include "include/helpers.hpp"
#include "include/probit_helpers.hpp"
#include "include/hs_helpers.hpp"

// **********************************************************************#
// individual updaters
// **********************************************************************#
// [[Rcpp::export]]
double probit_hs_elbo(
  Eigen::MatrixXd& X_s, Eigen::VectorXi& y_s, Rcpp::List& param_z,
  Rcpp::List& param_b0, Rcpp::List& param_b, Rcpp::List& param_lambda,
  Rcpp::List& param_xi, Rcpp::List& param_gamma, Rcpp::List& param_nu,
  int& N, int&  S, int& P
){
  // z
  double lp_m_lq_z = probit_lp_m_lq_z(X_s, y_s, param_b0, param_b, N, S);

  // ----- lp -----
  // b0
  double mu_prec0 = 0.000001;
  double mu_log_prec0 = -6 * std::log(10.0);
  double lp_b0 = lp_univ_normal(mu_prec0, mu_log_prec0, param_b0);

  // b
  double mu_lambda = param_lambda["mu"];
  Eigen::VectorXd mu_gamma = param_gamma["mu"];
  double mu_log_lambda = param_lambda["mu_log"];
  Eigen::VectorXd mu_log_gamma = param_gamma["mu_log"];

  Eigen::MatrixXd mu_prec = mu_lambda * mu_gamma.asDiagonal();
  double mu_logdet_prec = P * mu_log_lambda + mu_log_gamma.sum();
  double lp_b = lp_mv_normal(mu_prec, mu_logdet_prec, param_b, P);

  // lambda
  double mu_xi = param_xi["mu"];
  double mu_log_xi = param_xi["mu_log"];
  double lp_lambda = lp_univ_gamma(1.0/2.0, mu_xi, mu_log_xi, param_lambda);

  // xi
  double lp_xi = lp_univ_gamma(1.0/2.0, 1.0, 0.0, param_xi);

  // gamma
  Eigen::VectorXd a = 1.0 / 2.0 * Eigen::VectorXd::Constant(P, 1.0);
  Eigen::VectorXd mu_nu = param_nu["mu"];
  Eigen::VectorXd mu_log_nu = param_nu["mu_log"];
  double lp_gam = lp_gamma(a, mu_nu, mu_log_nu, param_gamma);

  // nu
  Eigen::VectorXd a1 = 1.0 / 2.0 * Eigen::VectorXd::Constant(P, 1.0);
  Eigen::VectorXd b1 = Eigen::VectorXd::Constant(P, 1.0);
  Eigen::VectorXd log_b1 = Eigen::VectorXd::Constant(P, 0.0);
  double lp_nu = lp_gamma(a1, b1, log_b1, param_nu);

  // ----- lq -----
  double lq_b0 = lq_univ_normal(param_b0);
  double lq_b = lq_mv_normal(param_b, P);
  double lq_lambda = lq_univ_gamma(param_lambda);
  double lq_xi = lq_univ_gamma(param_xi);
  double lq_gam = lq_gamma(param_gamma);
  double lq_nu = lq_gamma(param_nu);

  double elbo = lp_m_lq_z + lp_b0 + lp_b + lp_lambda + lp_xi + lp_gam +
    lp_nu - lq_b0 - lq_b - lq_lambda - lq_xi - lq_gam - lq_nu;

  return(elbo);
}

// **********************************************************************
// CAVI
// **********************************************************************
//' Univariate probit linear regression with a Horseshoe prior using the CAVI
//' algorithm.
//' @param y Vector or responses (N by 1)
//' @param X Matrix of predictors (N by P)
//' @param n_iter Max number of iterations to run the algorithm for (default =
//'   1000). A convergence warning is issues if the algorithm runs for the max
//'   number of iterations.
//' @param verbose True or False. Do you want to print messages along the way?
//' @param rel_tol Relative tolerance used for convergence. Convergence is
//'   assesed using the evidence lower bound (ELBO) changes relative to five
//'   iterations prior.
//' @param type Correlation structure of the regression coefficients. Use 0 for
//'   full correlation and 1 for independece assumption.
//' @export
// [[Rcpp::export]]
Rcpp::List probit_hs_cavi(
  Eigen::VectorXi y, Eigen::MatrixXd X, int n_iter = 1000, bool verbose = true,
  double tol = 0.0001, int type = 0
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
  Rcpp::List param_z = vi_init_mv_normal(S, 1);
  Rcpp::List param_b0 = vi_init_normal();
  Rcpp::List param_b = vi_init_mv_normal(P, type);
  Rcpp::List param_lambda = vi_init_gamma(1);
  Rcpp::List param_xi = vi_init_gamma(1);
  Rcpp::List param_gamma = vi_init_gamma(P);
  Rcpp::List param_nu = vi_init_gamma(P);
  Rcpp::List param_tau = vi_init_gamma(1);

  // for prior mean of b
  Eigen::MatrixXd mu_prior_mat(P, P);
  Eigen::VectorXd mu_gamma(P);
  double mu_lambda;

  // vectors for storage
  Eigen::VectorXd elbo = Eigen::VectorXd::Constant(n_iter, 1);
  double rhot = 1.0;

  // main loop of algorithm
  int iters = 0;
  for(int i = 0; i < n_iter; i++)
  {
    Rcpp::checkUserInterrupt();
    if(verbose && (i % 100 == 0)) {
      Rcpp::Rcout << "Done with Iteration " << i << " of " << n_iter << "\r";
    }

    // local params
    probit_vi_z(X, param_b0, param_b, S, param_z);
    vi_update_mv_normal(param_z, 1, 1.0);
    natural_to_canonical(param_z, "mv_normal_ind");
    canonical_transform_probit_trunc_norm(param_z, y, S);

    // b0
    probit_vi_b0(X, param_z, param_b, N, S, param_b0);
    vi_update_normal(param_b0, rhot);
    natural_to_canonical(param_b0, "normal");

    // b
    mu_gamma = param_gamma["mu"];
    mu_lambda = param_lambda["mu"];
    mu_prior_mat = mu_lambda * mu_gamma.asDiagonal();
    probit_vi_b(
      X, param_z, param_b0, mu_prior_mat, N, S, P, type, true, param_b
    );
    vi_update_mv_normal(param_b, type, rhot);
    if(type == 0) {
      natural_to_canonical(param_b, "mv_normal");
    } else {
      natural_to_canonical(param_b, "mv_normal_ind");
    }

    // lambda
    hs_vi_lambda(param_b, param_tau, param_gamma, param_xi, P, param_lambda);
    vi_update_gamma(param_lambda, rhot);
    natural_to_canonical(param_lambda, "univ_gamma");

    // xi
    hs_vi_xi(param_lambda, param_xi);
    vi_update_gamma(param_xi, rhot);
    natural_to_canonical(param_xi, "univ_gamma");

    // gamma
    hs_vi_gamma(param_b, param_tau, param_lambda, param_nu, P, param_gamma);
    vi_update_gamma(param_gamma, rhot);
    natural_to_canonical(param_gamma, "gamma");

    // nu
    hs_vi_nu(param_gamma, P, param_nu);
    vi_update_gamma(param_nu, rhot);
    natural_to_canonical(param_nu, "gamma");

    // elbo
    elbo(i) = probit_hs_elbo(
      X, y, param_z, param_b0, param_b, param_lambda, param_xi, param_gamma,
      param_nu, N, S, P
    );

    iters = iters + 1;
    if(i > 4)
    {
      // check if lower bound decreases
      if(elbo(i) < elbo(i - 1))
        std::cout << "LOWER BOUND DECREASES" << "\n";
      if((1.0 - elbo(i) / elbo(i - 5)) < tol) { break; }
      if(i == (n_iter - 1))
        std::cout << "VB DID NOT CONVERGE" << "\n";
    }
  }

  // values to rescale and return
  double mu_b0 = param_b0["mu"];
  double sigma2_b0 = param_b0["sigma2"];
  Eigen::VectorXd mu_b = param_b["mu"];
  Eigen::MatrixXd msigma_b = param_b["msigma"];
  double astar_lambda = param_lambda["shape"];
  double bstar_lambda = param_lambda["rate"];
  mu_gamma = param_gamma["mu"];

  // rescaled values - need mu_b0, sigma2_b0, msigma_b
  mu_b0 = mu_b0 - (mu_b.array() * s_x.array()).sum();
  sigma2_b0 = sigma2_b0 + s_x.transpose() * msigma_b * s_x;
  mu_b = mu_b.array() * vsigma_x_inv.array();
  msigma_b = vsigma_x_inv.asDiagonal() * msigma_b * vsigma_x_inv.asDiagonal();
  Eigen::VectorXd mu_z = param_z["mu"];

  Rcpp::List ret;
  ret["mu_z"] = mu_z;
  ret["mu_b0"] = mu_b0;
  ret["sigma2_b0"] = sigma2_b0;
  ret["mu_b"] = mu_b;
  ret["msigma_b"] = msigma_b;
  ret["astar_lambda"] = astar_lambda;
  ret["bstar_lambda"] = bstar_lambda;
  ret["mu_gamma"] = param_gamma["mu"];
  ret["elbo"] = elbo.topRows(iters);
  return(ret);
}

// **********************************************************************
// SVI
// **********************************************************************
//' Univariate probit linear regression with a ridge (normal) prior using the
//' SVI algorithm.
//' @param y Vector or responses (N by 1)
//' @param X Matrix of predictors (N by P)
//' @param n_iter Number of iterations to run the algorithm for (default =
//'   5000).
//' @param verbose True or False. Do you want to print messages along the way?
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
//'   \{0.5, 1\}}
//' @export
// [[Rcpp::export]]
Rcpp::List probit_hs_svi(
  Eigen::VectorXi y, Eigen::MatrixXd X, bool verbose = true, int n_iter = 1000,
  int type = 0, int batch_size = 10, double omega = 15.0, double kappa = 0.6,
  double const_rhot = 0.01
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
  Rcpp::List param_z = vi_init_mv_normal(S, 1);
  Rcpp::List param_b0 = vi_init_normal();
  Rcpp::List param_b = vi_init_mv_normal(P, type);
  Rcpp::List param_lambda = vi_init_gamma(1);
  Rcpp::List param_xi = vi_init_gamma(1);
  Rcpp::List param_gamma = vi_init_gamma(P);
  Rcpp::List param_nu = vi_init_gamma(P);
  Rcpp::List param_tau = vi_init_gamma(1);

  // for prior mean of b
  Eigen::MatrixXd mu_prior_mat(P, P);
  Eigen::VectorXd mu_gamma(P);
  double mu_lambda;

  // vectors for storage
  Eigen::VectorXd elbo = Eigen::VectorXd::Constant(n_iter, 1);

  // svi specific storage
  Eigen::VectorXi y_s(S);
  Eigen::MatrixXd X_s(S, P);
  Rcpp::IntegerVector the_sample = Rcpp::seq(0, S - 1);
  Rcpp::IntegerVector seq_samp = Rcpp::seq(0, N - 1);
  double rhot;

  // main loop of algorithm
  int iters = 0;
  for(int i = 0; i < n_iter; i++)
  {
    Rcpp::checkUserInterrupt();

    if(const_rhot <= 0)
      rhot = std::exp(-kappa * std::log(i + 1.0 + omega));
    else
      rhot = const_rhot;

    if(verbose && (i % 1000 == 0)) {
      Rcpp::Rcout << "Done with Iteration " << i <<
        " with step size " << rhot << "\r";
    }

    // ----- sample data -----
    the_sample = Rcpp::sample(seq_samp, S, false);
    get_subsample_probit(y, X, y_s, X_s, S, N, the_sample);

    // ----- local params -----
    probit_vi_z(X_s, param_b0, param_b, S, param_z);
    vi_update_mv_normal(param_z, 1, 1.0);
    natural_to_canonical(param_z, "mv_normal_ind");
    canonical_transform_probit_trunc_norm(param_z, y_s, S);

    // ----- global params -----

    // b0
    probit_vi_b0(X_s, param_z, param_b, N, S, param_b0);

    // b
    mu_gamma = param_gamma["mu"];
    mu_lambda = param_lambda["mu"];
    mu_prior_mat = mu_lambda * mu_gamma.asDiagonal();
    probit_vi_b(
      X_s, param_z, param_b0, mu_prior_mat, N, S, P, type, false, param_b
    );

    // hs specific params
    hs_vi_lambda(param_b, param_tau, param_gamma, param_xi, P, param_lambda);
    hs_vi_xi(param_lambda, param_xi);
    hs_vi_gamma(param_b, param_tau, param_lambda, param_nu, P, param_gamma);
    hs_vi_nu(param_gamma, P, param_nu);

    // ----- gradient step -----
    vi_update_normal(param_b0, rhot);
    vi_update_mv_normal(param_b, type, rhot);
    vi_update_gamma(param_lambda, rhot);
    vi_update_gamma(param_xi, rhot);
    vi_update_gamma(param_gamma, rhot);
    vi_update_gamma(param_nu, rhot);

    // ----- calculate means and variances for next iteration -----
    natural_to_canonical(param_b0, "normal");
    if(type == 0) {
      natural_to_canonical(param_b, "mv_normal");
    } else {
      natural_to_canonical(param_b, "mv_normal_ind");
    }
    natural_to_canonical(param_lambda, "univ_gamma");
    natural_to_canonical(param_xi, "univ_gamma");
    natural_to_canonical(param_gamma, "gamma");
    natural_to_canonical(param_nu, "gamma");

    // elbo
    elbo(i) = probit_hs_elbo(
      X_s, y_s, param_z, param_b0, param_b, param_lambda, param_xi, param_gamma,
      param_nu, N, S, P
    );
    iters = iters + 1;
  }

  // values to rescale and return
  double mu_b0 = param_b0["mu"];
  double sigma2_b0 = param_b0["sigma2"];
  Eigen::VectorXd mu_b = param_b["mu"];
  Eigen::MatrixXd msigma_b = param_b["msigma"];
  double astar_lambda = param_lambda["shape"];
  double bstar_lambda = param_lambda["rate"];
  mu_gamma = param_gamma["mu"];
  Eigen::VectorXd mu_z = param_z["mu"];

  // rescaled values - need mu_b0, sigma2_b0, msigma_b
  mu_b0 = mu_b0 - (mu_b.array() * s_x.array()).sum();
  sigma2_b0 = sigma2_b0 + s_x.transpose() * msigma_b * s_x;
  mu_b = mu_b.array() * vsigma_x_inv.array();
  msigma_b = vsigma_x_inv.asDiagonal() * msigma_b * vsigma_x_inv.asDiagonal();

  Rcpp::List ret;
  ret["mu_b0"] = mu_b0;
  ret["sigma2_b0"] = sigma2_b0;
  ret["mu_b"] = mu_b;
  ret["msigma_b"] = msigma_b;
  ret["astar_lambda"] = astar_lambda;
  ret["bstar_lambda"] = bstar_lambda;
  ret["mu_gamma"] = param_gamma["mu"];
  ret["elbo"] = elbo;
  return(ret);
}
