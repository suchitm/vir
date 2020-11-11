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
// samples the mean
double lm_lasso_gibbs_b0(double& y_bar, double& tau, int& N, double& b0)
{
    double G = (N * tau + 0.000001);
    double g = N * tau * y_bar;
    double mu = g / G;
    double sd = std::sqrt(1.0 / G);
    b0 = Rcpp::rnorm(1, mu, sd)(0);
    return(b0);
}

// function to update the coefficients in a linear regression
Eigen::VectorXd lm_lasso_gibbs_b(
    Eigen::MatrixXd& X, Eigen::VectorXd& y_tilde, Eigen::VectorXd& gammas,
    double& tau, int& P, Eigen::VectorXd& b
){
    Eigen::MatrixXd G = X.transpose() * X;
    G += gammas.asDiagonal();
    G = tau * G;
    Eigen::VectorXd g = tau * X.transpose() * y_tilde;
    Eigen::LLT<Eigen::MatrixXd> chol_G(G);
    Eigen::VectorXd mu(chol_G.solve(g));
    b = mu + chol_G.matrixU().solve(conv(Rcpp::rnorm(P, 0, 1)));
    return(b);
}

// updates the precision parameter tau in a bayesian lasso regression
double lm_lasso_gibbs_tau(
    Eigen::VectorXd& ehat, Eigen::VectorXd& b, Eigen::VectorXd& gammas,
    double& a_tau, double& b_tau, int& N, int& P, double& tau
){
    double shape = (N + P)/2.0 + a_tau;
    double rate = 1/2.0 * (
        ehat.squaredNorm() + b.transpose() * gammas.asDiagonal() * b
    ) + b_tau;
    tau = Rcpp::rgamma(1, shape, 1/rate)(0);
    return(tau);
}

// update the latent variables for the double exponential prior
Eigen::VectorXd lm_lasso_gibbs_gammas(
    double& tau, double& lambda2, Eigen::VectorXd& b, int& P,
    Eigen::VectorXd& gammas
){
    for(int p = 0; p < P; p++)
    {
        double mu = std::sqrt(lambda2 / (tau * b(p) * b(p)));
        gammas(p) = rinvgauss_cpp(1, mu, lambda2)(0);
    }
    return(gammas);
}

// updates the prior precision parameter of b, lambda, in a bayesian lasso
// regression
double lm_lasso_gibbs_lambda2(
    Eigen::VectorXd& gammas, double& a_lambda, double& b_lambda, int& P,
    double& lambda2
){
    double shape = P + a_lambda;
    double rate = 1/2.0 * gammas.array().inverse().sum() + b_lambda;
    lambda2 = Rcpp::rgamma(1, shape, 1/rate)(0);
    return(lambda2);
}

//**********************************************************************//
// Main function
//**********************************************************************//
//' Univariate normal linear regression with a LASSO (double-exponential) prior
//' using a Gibbs sampler.
//' @param y Vector or responses (N by 1)
//' @param X Matrix of predictors (N by P)
//' @param n_iter Number of iterations to run the algorithm for
//' @param verbose True or False. Do you want to print messages along the way?
//' @param a_tau Prior shape parameter for the likelihood precision.
//' @param b_tau Prior rate parameter for the likelihood precision.
//' @param a_lambda Prior shape parameter for the coefficient precision
//'   (shrinkage) term.
//' @param b_lambda Prior rate parameter for the coefficient precision
//'   (shrinkage) term.
//' @export
// [[Rcpp::export]]
Rcpp::List lm_lasso_gibbs(
    Eigen::VectorXd y, Eigen::MatrixXd X, bool verbose = true,
    int n_iter = 10000, double a_tau = 0.1, double b_tau = 0.1,
    double a_lambda = 0.1, double b_lambda = 0.1
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
    Eigen::MatrixXd gammas_mat = Eigen::MatrixXd::Constant(n_iter, P, 1.0);
    Eigen::VectorXd tau_vec = Eigen::VectorXd::Constant(n_iter, 1.0);
    Eigen::VectorXd lambda2_vec = Eigen::VectorXd::Constant(n_iter, 1.0);
    Eigen::VectorXd log_lik_vec = Eigen::VectorXd::Constant(n_iter, 1.0);

    // starting values
    double b0 = 0.0;
    double tau = 1.0;
    double lambda2 = 1.0;
    Eigen::VectorXd gammas = Eigen::VectorXd::Constant(P, 1.0);
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
        lm_lasso_gibbs_b0(y_bar, tau, N, b0);

        // sample coefficient vector
        lm_lasso_gibbs_b(X, y_tilde, gammas, tau, P, b);

        // update variance
        ehat = y - b0 * ones - X * b;
        lm_lasso_gibbs_tau(ehat, b, gammas, a_tau, b_tau, N, P, tau);

        // double exponential latent variables
        lm_lasso_gibbs_gammas(tau, lambda2, b, P, gammas);

        // global shrinkage parameter
        lm_lasso_gibbs_lambda2(gammas, a_lambda, b_lambda, P, lambda2);

        // store results
        b0_vec(i) = b0 - (b.array() * s_x.array()).sum();
        b_mat.row(i) = b.array() / vsigma_x.transpose().array();
        gammas_mat.row(i) = gammas;
        tau_vec(i) = tau;
        lambda2_vec(i) = lambda2;
        log_lik_vec(i) = N/2.0 * tau - tau/2.0 * ehat.squaredNorm();
    }
    List ret;
    ret["b0_vec"] = b0_vec;
    ret["b_mat"] = b_mat;
    ret["gammas_mat"] = gammas_mat;
    ret["tau_vec"] = tau_vec;
    ret["lambda2_vec"] = lambda2_vec;
    ret["log_lik_vec"] = log_lik_vec;
    return(ret);
}
