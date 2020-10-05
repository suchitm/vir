#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include "typedefs.h"
#include "include/helpers.hpp"

using namespace Eigen;
using namespace Rcpp;
using namespace std;

//' samples the mean
//' @export
// [[Rcpp::export]]
double lm_ridge_gibbs_b0(double& y_bar, double& tau, double& b0, int& N)
{
    double G = (N * tau + 0.000001);
    double g = N * tau * y_bar;
    double mu = g / G;
    double sd = std::sqrt(1.0 / G);
    b0 = Rcpp::rnorm(1, mu, sd)(0);
    return(b0);
}

//' function to update the coefficients in a linear regression
//' @param X N by P matrix of covariates
//' @export
// [[Rcpp::export]]
Eigen::VectorXd lm_ridge_gibbs_b(
    Eigen::MatrixXd& X, Eigen::VectorXd& y, double& lambda, double& tau,
    Eigen::VectorXd& b, int& P
){
    Eigen::MatrixXd G = tau * (X.transpose() * X +
       lambda * MatrixXd::Identity(P, P));
    Eigen::VectorXd g = tau * X.transpose() * y;
    Eigen::LLT<Eigen::MatrixXd> chol_G(G);
    Eigen::VectorXd mu(chol_G.solve(g));
    b = mu + chol_G.matrixU().solve(conv(Rcpp::rnorm(P, 0, 1)));
    return(b);
}

//' updates the precision parameter tau in a bayesian ridge regression
//' @export
// [[Rcpp::export]]
double lm_ridge_gibbs_tau(
    Eigen::VectorXd& ehat, Eigen::VectorXd& b, double& lambda,
    double& tau, double& a_tau, double& b_tau, int& N, int& P
){
    double shape = (N + P)/2.0 + a_tau;
    double rate = 1/2.0 * (
        ehat.squaredNorm() + lambda * b.squaredNorm()
    ) + b_tau;
    tau = Rcpp::rgamma(1, shape, 1/rate)(0);
    return(tau);
}

//' updates the prior precision parameter of b, lambda, in a bayesian ridge
//' regression
//' @export
// [[Rcpp::export]]
double lm_ridge_gibbs_lambda(
    Eigen::VectorXd& b, double& tau, double& lambda, double& a_lambda,
    double& b_lambda, int& P
){
    double shape = P/2.0 + a_lambda;
    double rate = tau/2.0 * b.squaredNorm() + b_lambda;
    lambda = Rcpp::rgamma(1, shape, 1/rate)(0);
    return(lambda);
}
