#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include "typedefs.h"
#include "include/tuvn_helpers.hpp"
#include "include/probit_helpers.hpp"
#include "include/helpers.hpp"

//**********************************************************************//
// other helpers
//**********************************************************************//
Eigen::VectorXd cum_prod(Eigen::VectorXd& x)
{
  int n = x.size();
  Eigen::VectorXd out_vec = Eigen::VectorXd::Constant(n, 1.0);
  double prod = 1.0;
  for(int i = 0; i < n; i++)
  {
    prod = prod * x(i);
    out_vec(i) = prod;
  }
  return(out_vec);
}

//**********************************************************************//
// individual samplers
//**********************************************************************//
Eigen::MatrixXd mv_probit_gibbs_Z(
  Eigen::MatrixXi& Y, Eigen::MatrixXd& Eta, double& tau, int& N, int& M,
  Eigen::MatrixXd& Z
){
  double this_sd = 1.0 / std::sqrt(tau);
  for(int m = 0; m < M; m++)
  {
    for(int n = 0; n < N; n++)
    {
      if(Y(n, m) == 1)
        Z(n, m) = rtuvn(1, Eta(n, m), this_sd, 0.0, R_PosInf)(0);
      else
        Z(n, m) = rtuvn(1, Eta(n, m), this_sd, R_NegInf, 0.0)(0);
    }
  }
  return(Z);
}

Eigen::MatrixXd mvlm_uninf_gibbs_mpsi(
  Eigen::MatrixXd& E_hat, Eigen::MatrixXd& mtheta, double& tau, int& N, int& K,
  Eigen::MatrixXd& mpsi
){
  Eigen::MatrixXd G = tau * mtheta.transpose() * mtheta +
    Eigen::MatrixXd::Identity(K, K);
  Eigen::LLT<Eigen::MatrixXd> chol_G(G);
  for(int n = 0; n < N; n++)
  {
    Eigen::VectorXd g = tau * mtheta.transpose() * E_hat.row(n).transpose();
    Eigen::VectorXd mu(chol_G.solve(g));
    mpsi.row(n) = mu + chol_G.matrixU().solve(conv(Rcpp::rnorm(K, 0, 1)));
  }
  return(mpsi);
}

Eigen::VectorXd mvlm_uninf_gibbs_b0(
  Eigen::MatrixXd& E_hat, double& tau, int& N, int& M, Eigen::VectorXd& b0
){
  double G = N * tau + 0.000001;
  for(int m = 0; m < M; m++)
  {
    double g = tau * E_hat.col(m).sum();
    double mu = g / G;
    double sd = 1.0 / std::sqrt(G);
    b0(m) = Rcpp::rnorm(1, mu, sd)(0);
  }
  return(b0);
}

Eigen::MatrixXd mvlm_uninf_gibbs_B(
  Eigen::MatrixXd& E_hat, Eigen::MatrixXd& X, double& tau, int& M,
  int& P, Eigen::MatrixXd& b
){
  Eigen::MatrixXd G = tau * X.transpose() * X +
    0.000001 * Eigen::MatrixXd::Identity(P, P);
  Eigen::LLT<Eigen::MatrixXd> chol_G(G);
  for(int m = 0; m < M; m++)
  {
    Eigen::VectorXd g = tau * X.transpose() * E_hat.col(m);
    Eigen::VectorXd mu(chol_G.solve(g));
    b.row(m) = mu + chol_G.matrixU().solve(conv(Rcpp::rnorm(P, 0, 1)));
  }
  return(b);
}

Eigen::MatrixXd mvlm_uninf_gibbs_mtheta(
  Eigen::MatrixXd& E_hat, Eigen::MatrixXd& mpsi, double& tau, int& M, int& K,
  Eigen::MatrixXd& mtheta
){
  Eigen::MatrixXd G = tau * mpsi.transpose() * mpsi +
    0.000001 * Eigen::MatrixXd::Identity(K, K);
  Eigen::LLT<Eigen::MatrixXd> chol_G(G);
  for(int m = 0; m < M; m++)
  {
    Eigen::VectorXd g = tau * mpsi.transpose() * E_hat.col(m);
    Eigen::VectorXd mu(chol_G.solve(g));
    mtheta.row(m) = mu + chol_G.matrixU().solve(conv(Rcpp::rnorm(K, 0, 1)));
  }
  return(mtheta);
}

double mvlm_uninf_gibbs_tau(
  Eigen::MatrixXd& E_hat, int& N, int& M, double& a_tau, double& b_tau,
  double& tau
){
  double shape = N / 2.0 * M + a_tau;
  double rate = b_tau + 1.0/2.0 * E_hat.array().square().sum();
  tau = Rcpp::rgamma(1, shape, 1/rate)(0);
  return(tau);
}
