#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include "typedefs.h"
#include "include/helpers.hpp"

using namespace Eigen;
using namespace Rcpp;
using namespace std;

//**********************************************************************//
// individual samplers
//**********************************************************************//
//' samples the mean
//' @export
// [[Rcpp::export]]
double lm_hs_gibbs_b0(double& y_bar, double& tau, int& N, double& b0)
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
Eigen::VectorXd lm_hs_gibbs_b(
    Eigen::MatrixXd& X, Eigen::VectorXd& y_tilde, double& tau, double& lambda,
    Eigen::VectorXd& gammas, int& P, Eigen::VectorXd& b
){
    Eigen::MatrixXd G = X.transpose() * X;
    G += lambda * gammas.asDiagonal();
    G = tau * G;
    Eigen::VectorXd g = tau * X.transpose() * y_tilde;
    Eigen::LLT<Eigen::MatrixXd> chol_G(G);
    Eigen::VectorXd mu(chol_G.solve(g));
    b = mu + chol_G.matrixU().solve(conv(Rcpp::rnorm(P, 0, 1)));
    return(b);
}

//' updates the precision parameter tau in a bayesian hs regression
//' @export
// [[Rcpp::export]]
double lm_hs_gibbs_tau(
    Eigen::VectorXd& ehat, Eigen::VectorXd& b, double& lambda,
    Eigen::VectorXd& gammas, double& a_tau, double& b_tau, int& N, int& P,
    double& tau
){
    double shape = (N + P) / 2.0 + a_tau;
    double rate = 1/2.0 * (
        ehat.squaredNorm() + lambda * b.transpose() * gammas.asDiagonal() * b
    ) + b_tau;
    tau = Rcpp::rgamma(1, shape, 1/rate)(0);
    return(tau);
}

//' @export
// [[Rcpp::export]]
double lm_hs_gibbs_lambda(
    Eigen::VectorXd& b, double& tau, Eigen::VectorXd& gammas, double& xi,
    int& P, double& lambda
){
    double shape = (P + 1) / 2.0;
    double rate = xi +
        tau / 2.0 * (gammas.array() * b.array().square()).sum();
    lambda = Rcpp::rgamma(1, shape, 1/rate)(0);
    return(lambda);
}

//' update the latent variables for the double exponential prior
// [[Rcpp::export]]
Eigen::VectorXd lm_hs_gibbs_gammas(
    double& tau, double& lambda, Eigen::VectorXd& nus, Eigen::VectorXd& b,
    int& P, Eigen::VectorXd& gammas
){
    for(int p = 0; p < P; p++)
    {
        double shape = 1.0;
        double rate = nus(p) + b(p) * b(p) * tau * lambda / 2.0;
        gammas(p) = Rcpp::rgamma(1, shape, 1/rate)(0);
    }
    return(gammas);
}

// [[Rcpp::export]]
double lm_hs_gibbs_xi(double& lambda, double& xi)
{
    xi = Rcpp::rgamma(1, 1.0, 1.0 / (1.0 + lambda))(0);
    return(xi);
}

// [[Rcpp::export]]
Eigen::VectorXd lm_hs_gibbs_nus(
    Eigen::VectorXd& gammas, int& P, Eigen::VectorXd& nus
){
    for(int p = 0; p < P; p++)
    {
        nus(p) = Rcpp::rgamma(1, 1.0, 1.0 / (1.0 + gammas(p)))(0);
    }
    return(nus);
}

//**********************************************************************//
// Main function
//**********************************************************************//
//' Univariate normal linear regression with a Horseshoe prior using a Gibbs
//' sampler.
//' @param y Vector or responses (N by 1)
//' @param X Matrix of predictors (N by P)
//' @param n_iter Number of iterations to run the algorithm for
//' @param verbose True of False. Do you want to print messages along the way?
//' @param a_tau Prior shape parameter for the likelihood precision.
//' @param b_tau Prior rate parameter for the likelihood precision.
//' @export
// [[Rcpp::export]]
Rcpp::List lm_hs_gibbs(
    Eigen::VectorXd y, Eigen::MatrixXd X, bool verbose = true,
    int n_iter = 10000, double a_tau = 0.1, double b_tau = 0.1
){
    int N = X.rows();
    int P = X.cols();
    Eigen::VectorXd ones = Eigen::VectorXd::Constant(N, 1);

    // center y
    double y_bar = y.mean();
    Eigen::VectorXd y_tilde = y - y_bar * ones;

    // scale X
    Eigen::RowVectorXd vmu_x = X.colwise().mean();
    Eigen::RowVectorXd vsigma_x =
        (X.rowwise() - vmu_x).colwise().squaredNorm() / (X.rows() - 1);
    vsigma_x = vsigma_x.array().sqrt();
    Eigen::VectorXd s_x = vmu_x.array() / vsigma_x.array();
    X = (X.rowwise() - vmu_x).array().rowwise() / vsigma_x.array();

    // initializing matricies to store results
    Eigen::VectorXd b0_vec = Eigen::VectorXd::Constant(n_iter, 1.0);
    Eigen::MatrixXd b_mat = Eigen::MatrixXd::Constant(n_iter, P, 1.0);
    Eigen::VectorXd tau_vec = Eigen::VectorXd::Constant(n_iter, 1.0);
    Eigen::VectorXd lambda_vec = Eigen::VectorXd::Constant(n_iter, 1.0);
    Eigen::VectorXd xi_vec = Eigen::VectorXd::Constant(n_iter, 1.0);
    Eigen::MatrixXd gammas_mat = Eigen::MatrixXd::Constant(n_iter, P, 1.0);
    Eigen::MatrixXd nus_mat = Eigen::MatrixXd::Constant(n_iter, P, 1.0);
    Eigen::VectorXd log_lik_vec = Eigen::VectorXd::Constant(n_iter, 1.0);

    // starting values
    double b0 = 0.0;
    double tau = 1.0;
    double lambda = 1.0;
    double xi = 1.0;
    Eigen::VectorXd gammas = Eigen::VectorXd::Constant(P, 1.0);
    Eigen::VectorXd nus = Eigen::VectorXd::Constant(P, 1.0);
    Eigen::VectorXd b;
    Eigen::VectorXd ehat;

    // main loop of sampler
    for(int i = 0; i < n_iter; i++)
    {
        // check interrupt and print progress
        Rcpp::checkUserInterrupt();
        if(verbose && (i % 1000 == 0)) {
            Rcout << "Done with Iteration " << i << " of " << n_iter << "\r";
        }

        // sample mean
        lm_hs_gibbs_b0(y_bar, tau, N, b0);

        // sample coefficient vector
        lm_hs_gibbs_b(X, y_tilde, tau, lambda, gammas, P, b);

        // update variance
        ehat = y - b0 * ones - X * b;
        lm_hs_gibbs_tau(ehat, b, lambda, gammas, a_tau, b_tau, N, P, tau);

        // global shrinkage parameter
        lm_hs_gibbs_lambda(b, tau, gammas, xi, P, lambda);

        // double exponential latent variables
        lm_hs_gibbs_gammas(tau, lambda, nus, b, P, gammas);

        // hyperpriors
        lm_hs_gibbs_xi(lambda, xi);
        lm_hs_gibbs_nus(gammas, P, nus);

        // store results
        b0_vec(i) = b0 - (b.array() * s_x.array()).sum();
        b_mat.row(i) = b.transpose().array() / vsigma_x.array();
        tau_vec(i) = tau;
        lambda_vec(i) = lambda;
        xi_vec(i) = xi;
        gammas_mat.row(i) = gammas;
        nus_mat.row(i) = nus;

        log_lik_vec(i) = N/2.0 * tau - tau/2.0 * ehat.squaredNorm();
    }
    List ret;
    ret["b0_vec"] = b0_vec;
    ret["b_mat"] = b_mat;
    ret["tau_vec"] = tau_vec;
    ret["lambda_vec"] = lambda_vec;
    ret["gammas_mat"] = gammas_mat;
    ret["nus_mat"] = nus_mat;
    ret["log_lik_vec"] = log_lik_vec;
    return(ret);
}
