#include <RcppEigen.h>
#include <Rmath.h>
// [[Rcpp::depends(RcppEigen)]]

#include "typedefs.h"
#include "include/helpers.hpp"

using namespace Eigen;
using Eigen::Map;

//*********************************************************
// function to convert numeric vector into Eigen::VectorXd
//*********************************************************
Eigen::VectorXd conv(Rcpp::NumericVector X)
{
    Eigen::Map<Eigen::VectorXd> XS(Rcpp::as<Eigen::Map<Eigen::VectorXd> >(X));
    return(XS);
}

// *************************************************************************
// funciton to create a sparse identity matrix. eigen creation not obvious
// *************************************************************************
SpMat sp_eye(int n)
{
    std::vector<T> tripletList;
    tripletList.reserve(n);
    for(int i = 0; i < n; i++)
    {
        tripletList.push_back(T(i, i, 1));
    }
    SpMat mat(n, n);
    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    return(mat);
}

// *************************************************************************
// drops columm for matrix; workaround to lack of eigen functionality
// *************************************************************************
Eigen::MatrixXd drop_column(Eigen::MatrixXd X, int index, int N, int P)
{
    Eigen::MatrixXd retX(N, P-1);
    retX << X.leftCols(index), X.rightCols(P - index - 1);
    return(retX);
}

// *************************************************************************
// drops index from vector; workaround to lack of eigen functionality
// *************************************************************************
Eigen::VectorXd drop_index(Eigen::VectorXd X, int index, int P)
{
    Eigen::VectorXd retX(P-1, 1);
    retX << X.head(index), X.tail(P - index - 1);
    return(retX);
}

Rcpp::NumericVector rinvgauss_cpp(int n, double mu, double lambda)
{
    Rcpp::Function f("rinvgauss");
    return f(Rcpp::_["n"] = 1, Rcpp::_["mean"] = mu, Rcpp::_["shape"] = lambda);
}

// *************************************************************************
// helper functions for SVB algorithms
// *************************************************************************
// -------------------------------------------------
// for initializing parameters
// -------------------------------------------------
Rcpp::List vi_init_normal()
{
  Rcpp::List param;
  param["mu"] = 0.0;
  param["sigma2"] = 1.0;
  param["delta1"] = 1.0;
  param["delta2"] = -1.0 / 2.0;
  param["delta1_t"] = 1.0;
  param["delta1_t"] = -1.0 / 2.0;
  return(param);
}

Rcpp::List vi_init_mv_normal(int& P, int type)
{
  Rcpp::List param;
  param["mu"] = Eigen::VectorXd::Constant(P, 0.0);
  param["msigma"] = Eigen::MatrixXd::Identity(P, P);
  param["vsigma2"] = Eigen::VectorXd::Constant(P, 1.0);
  param["delta1"] = Eigen::VectorXd::Constant(P, 0.0);
  param["delta1_t"] = Eigen::VectorXd::Constant(P, 0.0);
  if(type == 0)
  {
    param["delta2"] = -1.0 / 2.0 * Eigen::MatrixXd::Identity(P, P);
    param["delta2_t"] = -1.0 / 2.0 * Eigen::MatrixXd::Identity(P, P);
  }
  else
  {
    param["delta2"] = -1.0 / 2.0 * Eigen::VectorXd::Constant(P, 1.0);
    param["delta2_t"] = -1.0 / 2.0 * Eigen::VectorXd::Constant(P, 1.0);
  }
  return(param);
}

Rcpp::List vi_init_gamma(int P)
{
  Rcpp::List param;
  if(P == 1)
  {
    param["shape"] = 1.0;
    param["rate"] = 1.0;
    param["mu"] = 1.0;
    param["mu_log"] = -0.5772157;
    param["delta1"] = 0.0;
    param["delta2"] = -1.0;
    param["delta1_t"] = 0.0;
    param["delta2_t"] = -1.0;
  }
  else
  {
    param["shape"] = Eigen::VectorXd::Constant(P, 1.0);
    param["rate"] = Eigen::VectorXd::Constant(P, 1.0);
    param["mu"] = Eigen::VectorXd::Constant(P, 1.0);
    param["mu_log"] = Eigen::VectorXd::Constant(P, -0.5772157);
    param["delta1"] = Eigen::VectorXd::Constant(P, 0.0);
    param["delta2"] = Eigen::VectorXd::Constant(P, -1.0);
    param["delta1_t"] = Eigen::VectorXd::Constant(P, 0.0);
    param["delta2_t"] = Eigen::VectorXd::Constant(P, -1.0);
  }
  return(param);
}

Rcpp::List vi_init_inv_gauss(int& P)
{
  Rcpp::List param;
  param["mu"] = Eigen::VectorXd::Constant(P, 1.0);
  param["lambda"] = Eigen::VectorXd::Constant(P, 1.0);
  param["mu_inv"] = Eigen::VectorXd::Constant(P, 2.0);
  param["delta1"] = Eigen::VectorXd::Constant(P, -1.0/2.0);
  param["delta2"] = Eigen::VectorXd::Constant(P, -1.0/2.0);
  param["delta1_t"] = Eigen::VectorXd::Constant(P, -1.0/2.0);
  param["delta2_t"] = Eigen::VectorXd::Constant(P, -1.0/2.0);
  return(param);
}

Rcpp::List vi_init_indep_matrix_normal(int& n_rows, int& n_cols, double mu)
{
  Rcpp::List param;
  param["mu"] = Eigen::MatrixXd::Constant(n_rows, n_cols, mu);
  param["delta1"] = Eigen::MatrixXd::Constant(n_rows, n_cols, mu);
  param["delta1_t"] = Eigen::MatrixXd::Constant(n_rows, n_cols, mu);
  param["vsigma2_mat"] = Eigen::MatrixXd::Constant(n_rows, n_cols, 1.0);
  Eigen::MatrixXd msigma_mat(n_rows * n_cols, n_cols);
  Eigen::MatrixXd delta2(n_rows * n_cols, n_cols);

  for(int n = 0; n < n_rows; n++)
  {
    msigma_mat.block(n * n_cols, 0, n_cols, n_cols) =
      Eigen::MatrixXd::Identity(n_cols, n_cols);
    delta2.block(n * n_cols, 0, n_cols, n_cols) =
      -1.0 / 2.0 * Eigen::MatrixXd::Identity(n_cols, n_cols);
  }
  param["msigma_mat"] = msigma_mat;
  param["delta2"] = delta2;
  param["delta2_t"] = delta2;
  return(param);
}

Rcpp::List vi_init_indep_matrix_gamma(int& n_rows, int& n_cols)
{
  Rcpp::List param;
  param["shape"] = Eigen::MatrixXd::Constant(n_rows, n_cols, 1.0);
  param["rate"] = Eigen::MatrixXd::Constant(n_rows, n_cols, 1.0);
  param["mu"] = Eigen::MatrixXd::Constant(n_rows, n_cols, 1.0);
  param["mu_log"] = Eigen::MatrixXd::Constant(n_rows, n_cols, -0.5772157);
  param["delta1"] = Eigen::MatrixXd::Constant(n_rows, n_cols, 0.0);
  param["delta2"] = Eigen::MatrixXd::Constant(n_rows, n_cols, -1.0);
  param["delta1_t"] = Eigen::MatrixXd::Constant(n_rows, n_cols, 0.0);
  param["delta2_t"] = Eigen::MatrixXd::Constant(n_rows, n_cols, -1.0);
  return(param);
}

//---------------------------------------------------
// get parameters needed for next iterations
//---------------------------------------------------
// delta_t -> delta
Rcpp::List vi_update_normal(Rcpp::List& param, double rhot)
{
  double delta1 = param["delta1"];
  double delta2 = param["delta2"];
  double delta1_t = param["delta1_t"];
  double delta2_t = param["delta2_t"];

  param["delta1"] = (1 - rhot) * delta1 + rhot * delta1_t;
  param["delta2"] = (1 - rhot) * delta2 + rhot * delta2_t;
  return(param);
}

Rcpp::List vi_update_mv_normal(Rcpp::List& param, int type, double rhot)
{
  Eigen::VectorXd delta1 = param["delta1"];
  Eigen::VectorXd delta1_t = param["delta1_t"];
  param["delta1"] = (1 - rhot) * delta1 + rhot * delta1_t;

  if(type == 0) {
    Eigen::MatrixXd delta2 = param["delta2"];
    Eigen::MatrixXd delta2_t = param["delta2_t"];
    param["delta2"] = (1 - rhot) * delta2 + rhot * delta2_t;
  } else {
    Eigen::VectorXd delta2 = param["delta2"];
    Eigen::VectorXd delta2_t = param["delta2_t"];
    param["delta2"] = (1 - rhot) * delta2 + rhot * delta2_t;
  }

  return(param);
}

Rcpp::List vi_update_gamma(Rcpp::List& param, double rhot)
{
  Eigen::VectorXd delta1 = param["delta1"];
  Eigen::VectorXd delta1_t = param["delta1_t"];
  Eigen::VectorXd delta2 = param["delta2"];
  Eigen::VectorXd delta2_t = param["delta2_t"];

  param["delta1"] = (1 - rhot) * delta1 + rhot * delta1_t;
  param["delta2"] = (1 - rhot) * delta2 + rhot * delta2_t;
  return(param);
}

Rcpp::List vi_update_inv_gauss(Rcpp::List& param, double rhot)
{
  Eigen::VectorXd delta1 = param["delta1"];
  Eigen::VectorXd delta1_t = param["delta1_t"];
  Eigen::VectorXd delta2 = param["delta2"];
  Eigen::VectorXd delta2_t = param["delta2_t"];

  param["delta1"] = (1 - rhot) * delta1 + rhot * delta1_t;
  param["delta2"] = (1 - rhot) * delta2 + rhot * delta2_t;
  return(param);
}

Rcpp::List vi_update_indep_matrix_normal(Rcpp::List& param, double rhot)
{
  Eigen::MatrixXd delta1 = param["delta1"];
  Eigen::MatrixXd delta1_t = param["delta1_t"];
  Eigen::MatrixXd delta2 = param["delta2"];
  Eigen::MatrixXd delta2_t = param["delta2_t"];

  param["delta1"] = (1 - rhot) * delta1 + rhot * delta1_t;
  param["delta2"] = (1 - rhot) * delta2 + rhot * delta2_t;
  return(param);
}

Rcpp::List vi_update_indep_matrix_gamma(Rcpp::List& param, double rhot)
{
  Eigen::MatrixXd delta1 = param["delta1"];
  Eigen::MatrixXd delta1_t = param["delta1_t"];
  Eigen::MatrixXd delta2 = param["delta2"];
  Eigen::MatrixXd delta2_t = param["delta2_t"];

  param["delta1"] = (1 - rhot) * delta1 + rhot * delta1_t;
  param["delta2"] = (1 - rhot) * delta2 + rhot * delta2_t;
  return(param);
}

// natural to canonical
Rcpp::List natural_to_canonical(Rcpp::List& param, std::string dist_type)
{
  if(dist_type == "normal")
  {
    double delta1 = param["delta1"];
    double delta2 = param["delta2"];
    param["mu"] = -1.0 / 2.0 * delta1 / delta2;
    param["sigma2"] = -1.0 / 2.0 / delta2;
  }
  else if (dist_type == "mv_normal")
  {
    Eigen::VectorXd g = param["delta1"];
    Eigen::MatrixXd G = param["delta2"];
    int P = G.rows();

    Eigen::LLT<Eigen::MatrixXd> chol_G(-2.0 * G);
    Eigen::MatrixXd msigma = chol_G.solve(MatrixXd::Identity(P, P));

    param["mu"] = msigma * g;
    param["msigma"] = msigma;
    param["vsigma2"] = msigma.diagonal();
    param["logdet_msigma"] = -2.0 *
        chol_G.matrixL().toDenseMatrix().diagonal().array().log().sum();
  }
  else if (dist_type == "mv_normal_ind")
  {
    Eigen::VectorXd delta1 = param["delta1"];
    Eigen::VectorXd delta2 = param["delta2"];

    Eigen::VectorXd vsigma2 = -1.0/2.0 / delta2.array();
    Eigen::VectorXd mu = delta1.array() * vsigma2.array();
    Eigen::MatrixXd msigma = vsigma2.asDiagonal();

    param["mu"] = mu;
    param["vsigma2"] = vsigma2;
    param["msigma"] = msigma;
    param["logdet_msigma"] = vsigma2.array().log().sum();
  }
  else if (dist_type == "matrix_normal_ind")
  {
    Eigen::MatrixXd delta1 = param["delta1"];
    Eigen::MatrixXd delta2 = param["delta2"];
    int n_rows = delta1.rows();
    int n_cols = delta1.cols();

    Eigen::MatrixXd mu(n_rows, n_cols);
    Eigen::MatrixXd vsigma2 (n_rows, n_cols);
    Eigen::MatrixXd msigma(n_rows * n_cols, n_cols);
    Eigen::VectorXd g(n_cols);
    Eigen::MatrixXd G(n_cols, n_cols);
    Eigen::MatrixXd temp_msigma(n_cols, n_cols);

    for(int n = 0; n < n_rows; n++)
    {
      g = delta1.row(n).transpose();
      G = delta2.block(n * n_cols, 0, n_cols, n_cols);

      Eigen::LLT<Eigen::MatrixXd> chol_G(-2.0 * G);
      temp_msigma = chol_G.solve(MatrixXd::Identity(n_cols, n_cols));

      mu.row(n) = temp_msigma * g;
      msigma.block(n * n_cols, 0, n_cols, n_cols) = temp_msigma;
      vsigma2.row(n) = temp_msigma.diagonal();
    }

    param["mu"] = mu;
    param["msigma_mat"] = msigma;
    param["vsigma2_mat"] = vsigma2;
  }
  else if (dist_type == "univ_gamma")
  {
    double delta1 = param["delta1"];
    double delta2 = param["delta2"];
    double shape = delta1 + 1;
    double rate = -1.0 * delta2;
    double mu = shape / rate;
    double mu_log = Rf_digamma(shape) - log(rate);

    param["shape"] = shape;
    param["rate"] = rate;
    param["mu"] = mu;
    param["mu_log"] = mu_log;
  }
  else if (dist_type == "gamma")
  {
    Eigen::VectorXd delta1 = param["delta1"];
    Eigen::VectorXd delta2 = param["delta2"];
    Eigen::VectorXd shape = delta1.array() + 1;
    Eigen::VectorXd rate = -1.0 * delta2;
    Eigen::VectorXd mu = shape.array() / rate.array();

    int P = rate.size();
    Eigen::VectorXd mu_log(P);
    for(int p = 0; p < P; p++)
    {
      mu_log(p) = Rf_digamma(shape(p)) - log(rate(p));
    }

    param["shape"] = shape;
    param["rate"] = rate;
    param["mu"] = mu;
    param["mu_log"] = mu_log;
  }
  else if (dist_type == "matrix_gamma_ind")
  {
    Eigen::MatrixXd delta1 = param["delta1"];
    Eigen::MatrixXd delta2 = param["delta2"];
    Eigen::MatrixXd shape = delta1.array() + 1.0;
    Eigen::MatrixXd rate = -1.0 * delta2;
    Eigen::MatrixXd mu = shape.array() / rate.array();

    int n_rows = rate.rows();
    int n_cols = rate.cols();
    Eigen::MatrixXd mu_log(n_rows, n_cols);
    for(int m = 0; m < n_rows; m++)
    {
      for(int p = 0; p < n_cols; p++)
      {
        mu_log(m, p) = Rf_digamma(shape(m, p)) - log(rate(m, p));
      }
    }

    param["shape"] = shape;
    param["rate"] = rate;
    param["mu"] = mu;
    param["mu_log"] = mu_log;
  }
  else if (dist_type == "inv_gauss")
  {
    Eigen::VectorXd delta1 = param["delta1"];
    Eigen::VectorXd delta2 = param["delta2"];
    Eigen::VectorXd mu = (delta2.array() / delta1.array()).sqrt();
    Eigen::VectorXd lambda = -2 * delta2;
    Eigen::VectorXd mu_inv = mu.array().inverse() + lambda.array().inverse();

    param["mu"] = mu;
    param["lambda"] = lambda;
    param["mu_inv"] = mu_inv;
  }
  else
  {
    Rcpp::Rcout << "NO VALID DISTRIBUTION" << std::endl;
  }
  return(param);
}

// subsample the data for SVI algorithm
void get_subsample_mvlm(
  Eigen::MatrixXd& Y, Eigen::MatrixXd& X, Eigen::MatrixXd& Y_s,
  Eigen::MatrixXd& X_s, int& S, int& N, Rcpp::IntegerVector the_sample
){
  // Rcpp::IntegerVector the_sample = sample_int(S, 0, N - 1);
  int this_samp;
  for(int s = 0; s < S; s++)
  {
    this_samp = the_sample(s);
    X_s.row(s) = X.row(this_samp);
    Y_s.row(s) = Y.row(this_samp);
  }
}

// subsample the data for SVI algorithm
void get_subsample_lm(
  Eigen::VectorXd& y, Eigen::MatrixXd& X, Eigen::VectorXd& y_s,
  Eigen::MatrixXd& X_s, int& S, int& N, Rcpp::IntegerVector the_sample
){
  int this_samp;
  for(int s = 0; s < S; s++)
  {
    this_samp = the_sample(s);
    X_s.row(s) = X.row(this_samp);
    y_s(s) = y(this_samp);
  }
}

// overload this function to allow integer y
void get_subsample_probit(
  Eigen::VectorXi& y, Eigen::MatrixXd& X, Eigen::VectorXi& y_s,
  Eigen::MatrixXd& X_s, int& S, int& N, Rcpp::IntegerVector the_sample
){
  int this_samp;
  for(int s = 0; s < S; s++)
  {
    this_samp = the_sample(s);
    X_s.row(s) = X.row(this_samp);
    y_s(s) = y(this_samp);
  }
}

void canonical_transform_probit_trunc_norm(
  Rcpp::List& param_z, Eigen::VectorXi& y, int& S
){
  Eigen::VectorXd eta = param_z["mu"];
  Eigen::VectorXd eta_new(S);

  double phi, Phi;
  for(int s = 0; s < S; s++)
  {
    Rcpp::checkUserInterrupt();
    phi = R::dnorm(-1 * eta(s), 0.0, 1.0, false);
    Phi = R::pnorm(-1 * eta(s), 0.0, 1.0, true, false);
    if(y(s) == 1) {
      eta_new(s) = eta(s) + phi / (1.0 - Phi);
    } else {
      eta_new(s) = eta(s) - phi / Phi;
    }
  }
  param_z["mu"] = eta_new;
}

void canonical_transform_mv_probit_trunc_norm(
  Rcpp::List& param_z, Eigen::MatrixXi& Y, int& S, int& M
){
  Eigen::MatrixXd eta = param_z["mu"];
  Eigen::MatrixXd eta_new(S, M);

  double phi, Phi;
  for(int m = 0; m < M; m++)
  {
    for(int s = 0; s < S; s++)
    {
      Rcpp::checkUserInterrupt();
      phi = R::dnorm(-1 * eta(s, m), 0.0, 1.0, false);
      Phi = R::pnorm(-1 * eta(s, m), 0.0, 1.0, true, false);
      if(Y(s, m) == 1) {
        eta_new(s, m) = eta(s, m) + phi / (1.0 - Phi);
      } else {
        eta_new(s, m) = eta(s, m) - phi / Phi;
      }
    }
  }
  param_z["mu"] = eta_new;
}

// *****************************************************************************
// ELBO helpers
// *****************************************************************************
double lm_log_lik(
  Eigen::MatrixXd& X_s, Eigen::VectorXd& y_s, Rcpp::List& param_b0,
  Rcpp::List& param_b, Rcpp::List& param_tau, int& N, int& S
){
  double mu_b0 = param_b0["mu"];
  double sigma2_b0 = param_b0["sigma2"];
  Eigen::VectorXd mu_b = param_b["mu"];
  Eigen::MatrixXd msigma_b = param_b["msigma"];
  double astar_tau = param_tau["shape"];
  double bstar_tau = param_tau["rate"];
  double mu_tau = param_tau["mu"];

  double ll = -N / 2.0 * std::log(2.0 * M_PI) +
    N / 2.0 * (Rf_digamma(astar_tau) - std::log(bstar_tau)) -
    N * 1.0 * mu_tau / 2.0 / S * (
      (y_s - mu_b0 * Eigen::VectorXd::Ones(S) - X_s * mu_b).squaredNorm() +
      S * sigma2_b0 +
      (X_s.transpose() * X_s * msigma_b).trace()
    );

  return(ll);
}

double probit_lp_m_lq_z(
  Eigen::MatrixXd& X_s, Eigen::VectorXi& y_s, Rcpp::List& param_b0,
  Rcpp::List& param_b, int& N, int& S
){
  double mu_b0 = param_b0["mu"];
  double sigma2_b0 = param_b0["sigma2"];
  Eigen::VectorXd mu_b = param_b["mu"];
  Eigen::MatrixXd msigma_b = param_b["msigma"];
  Eigen::VectorXd eta = mu_b0 * Eigen::VectorXd::Constant(S, 1.0) + X_s * mu_b;

  double log_Phi, log1mexp;
  double sum_calc = 0.0;
  for(int s = 0; s < S; s++)
  {
    log_Phi = R::pnorm(-1 * eta(s), 0.0, 1.0, true, true);

    // need to compute log(1 - exp(log_Phi)) in a stable way
    if(log_Phi > -M_LN2)
      log1mexp = std::log(-expm1(log_Phi));
    else
      log1mexp = log1p(-std::exp(log_Phi));

    sum_calc += y_s(s) * log1mexp + (1 - y_s(s)) * log_Phi;
  }

  double lp_m_lq =
    - N / 2.0 * sigma2_b0 -
    N / 2.0 / S * (X_s.transpose() * X_s * msigma_b).trace() +
    N / 1.0 / S * sum_calc;

  return(lp_m_lq);
}

double lp_univ_normal(double& mu_prec, double& mu_log_prec, Rcpp::List& param)
{
  double mu = param["mu"];
  double sigma2 = param["sigma2"];

  double lp =
    -1.0 / 2.0 * std::log(2.0 * M_PI) +
    1.0 / 2.0 * mu_log_prec -
    mu_prec / 2.0 * (mu * mu + sigma2);

  return(lp);
}

double lp_mv_normal(
  Eigen::MatrixXd& mu_prec, double& mu_logdet_prec, Rcpp::List& param,
  int& P
){
  Eigen::VectorXd mu_b = param["mu"];
  Eigen::MatrixXd msigma_b = param["msigma"];

  double lp =
    -P / 2.0 * std::log(2.0 * M_PI) +
    1.0 / 2.0 * mu_logdet_prec -
    1.0 / 2.0 * (
      mu_b.transpose() * mu_prec * mu_b +
      (mu_prec * msigma_b).trace()
    );

  return(lp);
}

double lp_univ_gamma(
  double a, double mu_b, double mu_log_b, Rcpp::List& param
){
  double mu = param["mu"];
  double shape = param["shape"];
  double rate = param["rate"];

  double lp =
    a * mu_log_b -
    Rf_lgammafn(a) +
    (a - 1) * (Rf_digamma(shape) - std::log(rate)) -
    mu_b * mu;

  return(lp);
}

double lp_gamma(
  Eigen::VectorXd& a, Eigen::VectorXd& mu_b, Eigen::VectorXd& mu_log_b,
  Rcpp::List& param
){
  Eigen::VectorXd mu = param["mu"];
  Eigen::VectorXd mu_log = param["mu_log"];
  int P = mu_log.size();
  double lp = 0.0;

  for(int p = 0; p < P; p++)
  {
    lp += a(p) * mu_log_b(p) -
      Rf_lgammafn(a(p)) +
      (a(p) - 1) * mu_log(p) -
      mu_b(p) * mu(p);
  }

  return(lp);
}

double lq_univ_normal(Rcpp::List& param)
{
  double sigma2 = param["sigma2"];
  double lq = -1.0 / 2.0 * std::log(sigma2) -
    1.0 / 2.0 * (1 + std::log(2.0 * M_PI));
  return(lq);
}

double lq_mv_normal(Rcpp::List& param, int& P)
{
  double logdet_msigma = param["logdet_msigma"];
  double lq = -1.0 / 2.0 * logdet_msigma - P / 2.0 * (1 + std::log(2.0 * M_PI));
  return(lq);
}

double lq_univ_gamma(Rcpp::List& param)
{
  double shape = param["shape"];
  double rate = param["rate"];

  double lq =
    -Rf_lgammafn(shape) +
    (shape - 1) * Rf_digamma(shape) +
    std::log(rate) -
    shape;
  return(lq);
}

double lq_gamma(Rcpp::List& param)
{
  Eigen::VectorXd shape = param["shape"];
  Eigen::VectorXd rate = param["rate"];
  int P = shape.size();
  double lq = 0.0;
  double this_shape, this_rate;

  for(int p = 0; p < P; p++)
  {
    this_shape = shape(p);
    this_rate = rate(p);
    lq += -Rf_lgammafn(this_shape) +
      (this_shape - 1) * Rf_digamma(this_shape) +
      std::log(this_rate) -
      this_shape;
  }
  return(lq);
}

double lq_lasso_gamma(Rcpp::List& param_gamma)
{
  Eigen::VectorXd lambda = param_gamma["lambda"];
  Eigen::VectorXd mu = param_gamma["mu"];
  Eigen::VectorXd mu_inv = param_gamma["mu_inv"];
  double lq_gamma = (
    1.0 / 2.0 * lambda.array().log() -
    1.0 / 2.0 * std::log(2 * M_PI) -
    lambda.array() / (2.0 * mu.array()) -
    lambda.array() / 2.0 * mu_inv.array() +
    lambda.array() / mu.array()
  ).sum();
  return(lq_gamma);
}

double lp_lasso_gamma(Rcpp::List& param_lambda2, Rcpp::List& param_gamma)
{
  Eigen::VectorXd gamma_mu_inv = param_gamma["mu_inv"];
  double shape = param_lambda2["shape"];
  double rate = param_lambda2["rate"];
  double mu = param_lambda2["mu"];

  double lp_gamma = (
    Rf_digamma(shape) - std::log(rate) - std::log(2.0) -
    mu / 2.0 * gamma_mu_inv.array()
  ).sum();

  return(lp_gamma);
}









