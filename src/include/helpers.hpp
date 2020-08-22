#ifndef __HELPERS__
#define __HELPERS__

#include "../typedefs.h"

Eigen::VectorXd conv(Rcpp::NumericVector X);
SpMat sp_eye(int n);
Eigen::MatrixXd drop_column(Eigen::MatrixXd X, int index, int N, int P);
Eigen::VectorXd drop_index(Eigen::VectorXd X, int index, int P);
Rcpp::NumericVector rinvgauss_cpp(int n, double mu, double lambda);
void get_subsample_mvlm(
  Eigen::MatrixXd& Y, Eigen::MatrixXd& X, Eigen::MatrixXd& Y_s,
  Eigen::MatrixXd& X_s, int& S, int& N, Rcpp::IntegerVector the_sample
);

// susample for svi in univariate linear models
void get_subsample_lm(
  Eigen::VectorXd& y, Eigen::MatrixXd& X, Eigen::VectorXd& y_s,
  Eigen::MatrixXd& X_s, int& S, int& N, Rcpp::IntegerVector the_sample
);
void get_subsample_probit(
  Eigen::VectorXi& y, Eigen::MatrixXd& X, Eigen::VectorXi& y_s,
  Eigen::MatrixXd& X_s, int& S, int& N, Rcpp::IntegerVector the_sample
);

// vi inits
Rcpp::List vi_init_normal();
Rcpp::List vi_init_mv_normal(int& P, int type);
Rcpp::List vi_init_gamma(int P);
Rcpp::List vi_init_inv_gauss(int& P);
Rcpp::List vi_init_indep_matrix_normal(int& n_rows, int& n_cols, double mu);
Rcpp::List vi_init_indep_matrix_gamma(int& n_rows, int& n_cols);

// vi updates
Rcpp::List vi_update_normal(Rcpp::List& param, double rhot);
Rcpp::List vi_update_mv_normal(Rcpp::List& param, int type, double rhot);
Rcpp::List vi_update_gamma(Rcpp::List& param, double rhot);
Rcpp::List vi_update_inv_gauss(Rcpp::List& param, double rhot);
Rcpp::List vi_update_indep_matrix_normal(Rcpp::List& param, double rhot);
Rcpp::List vi_update_indep_matrix_gamma(Rcpp::List& param, double rhot);

// natural to canonical
Rcpp::List natural_to_canonical(Rcpp::List& param, std::string dist_type);
void canonical_transform_probit_trunc_norm(
  Rcpp::List& param_z, Eigen::VectorXi& y, int& S
);

// *****************************************************************************
// ELBO helpers
// *****************************************************************************
double lm_log_lik(
  Eigen::MatrixXd& X_s, Eigen::VectorXd& y_s, Rcpp::List& param_b0, 
  Rcpp::List& param_b, Rcpp::List& param_tau, int& N, int& S
);
double probit_lp_m_lq_z(
  Eigen::MatrixXd& X_s, Eigen::VectorXi& y_s, Rcpp::List& param_b0,
  Rcpp::List& param_b, int& N, int& S
);
double lp_univ_normal(double& mu_prec, double& mu_log_prec, Rcpp::List& param);
double lp_mv_normal(
  Eigen::MatrixXd& mu_prec, double& mu_logdet_prec, Rcpp::List& param,
  int& P
);
double lp_univ_gamma(double a, double mu_b, double mu_log_b, Rcpp::List& param);
double lp_gamma(
  Eigen::VectorXd& a, Eigen::VectorXd& mu_b, Eigen::VectorXd& mu_log_b,
  Rcpp::List& param
);
double lp_lasso_gamma(Rcpp::List& param_lambda2, Rcpp::List& param_gamma);

double lq_univ_normal(Rcpp::List& param);
double lq_mv_normal(Rcpp::List& param, int& P);
double lq_univ_gamma(Rcpp::List& param);
double lq_gamma(Rcpp::List& param);
double lq_lasso_gamma(Rcpp::List& param_gamma);

#endif // __HELPERS__
