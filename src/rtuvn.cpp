#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

#include "include/tuvn_helpers.hpp"

using namespace Eigen;
using namespace Rcpp;
using namespace std;

// Univariate Truncated Normal Distribution
//
// Random number generator for the truncated normal distribution. To
//     calculate the density please use dtuvn.
//
// @param n number of samples
// @param mean mean
// @param sd standard deviation
// @param lower lower bound
// @param upper upper bound
// @return a vector of generated samples
//
// @references Li, Y., & Ghosh, S. K. (2015). Efficient sampling methods for
//     truncated multivariate normal and student-t distributions subject to
//     linear inequality constraints. Journal of Statistical Theory and
//     Practice, 9(4), 712-732.
//
// @examples
// # sample from truncated normal with mean 10, sd 20, and lower = 10, upper = 20
// rtuvn(n = 1, mean = 10, sd = 20, lower = 10, upper = 20)
//
Eigen::VectorXd rtuvn(int n, double mean, double sd, double lower, double upper)
{
    if(lower > upper)
    {
        throw std::range_error("Error in rtuvn: lower is greater than upper");
    }
    // transform the boundaries
    double a = (lower - mean) / sd;
    double b = (upper - mean) / sd;

    // generate sample from TN(0, 1; a, b)
    Eigen::VectorXd Z(n);
    for(int i = 0; i < n; i++)
        Z(i) = sample_tuvsn(a, b);

    // transform the data back
    Z = sd * Z + Eigen::VectorXd::Constant(n, 1.0) * mean;
    return(Z);
}

