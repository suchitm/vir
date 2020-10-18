#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include "typedefs.h"
#include "include/helpers.hpp"
#include "include/ridge_updaters.hpp"

using namespace Eigen;
using namespace Rcpp;
using namespace std;

//' Univariate normal linear regression with a ridge (normal) prior using a
//' Gibbs sampler.
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
Rcpp::List lm_ridge_gibbs(
    Eigen::VectorXd y, Eigen::MatrixXd X, bool verbose = true,
    int n_iter = 10000, double a_tau = 0.01, double b_tau = 0.01,
    double a_lambda = 0.01, double b_lambda = 0.01
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
    Eigen::VectorXd b0_vec = Eigen::VectorXd::Constant(n_iter, 1);
    Eigen::MatrixXd b_mat = Eigen::MatrixXd::Constant(n_iter, P, 1);
    Eigen::VectorXd tau_vec = Eigen::VectorXd::Constant(n_iter, 1);
    Eigen::VectorXd lambda_vec = Eigen::VectorXd::Constant(n_iter, 1);
    Eigen::VectorXd log_lik_vec = Eigen::VectorXd::Constant(n_iter, 1);

    // starting values
    double b0 = 0.0;
    double tau = 1.0;
    double lambda = 1.0;
    Eigen::VectorXd b;
    Eigen::VectorXd ehat;

    // main loop of sampler
    for(int i = 0; i < n_iter; i++)
    {
        // check interrupt and print progress
        Rcpp::checkUserInterrupt();
        if(verbose && (i % 1000 == 0)) {
            Rcout << "Done with Iteration " << i << " of " << n_iter << "\n";
        }

        // sample the mean
        lm_ridge_gibbs_b0(y_bar, tau, b0, N);

        // update the coefs - b
        lm_ridge_gibbs_b(X, y_tilde, lambda, tau, b, P);

        // update tau
        ehat = y - b0 * ones - X * b;
        lm_ridge_gibbs_tau(ehat, b, lambda, tau, a_tau, b_tau, N, P);

        // update lambda
        lm_ridge_gibbs_lambda(b, tau, lambda, a_lambda, b_lambda, P);

        // store results
        b0_vec(i) = b0 - (b.array() * s_x.array()).sum();
        b_mat.row(i) = b.array() / vsigma_x.transpose().array();
        tau_vec(i) = tau;
        lambda_vec(i) = lambda;
        log_lik_vec(i) = N/2.0 * tau - tau/2.0 * ehat.squaredNorm();
    }
    List ret;
    ret["b0_vec"] = b0_vec;
    ret["b_mat"] = b_mat;
    ret["tau_vec"] = tau_vec;
    ret["lambda_vec"] = lambda_vec;
    ret["log_lik_vec"] = log_lik_vec;
    return(ret);
}
