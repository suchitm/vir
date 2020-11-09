#' Run a Gibbs sampler for the multivariate probit model. Uses the
#' multiplicative gamma process for the factor structure
#' @title Multivariate linear regression with a factor model
#' @param Y matrix of responses
#' @param X matrix of predictors to control for
#' @param K number of factors in the factor model
#' @param n_iter number of iterations to run the Gibbs sampler
#' @param verbose True or False. Print status of the sampler.
#' @return A list with samples for the intercept term (\code{b0_mat}),
#'   regression coefficients (\code{B_array}), and upper triangular elements of
#'   the correlation matrix (\code{cor_mat_upper_tri})
#' @export
#
# sim_dat = sim_mv_lm(20, 4, 3, 2)
# X = sim_dat$X
# Y = sim_dat$Y
# K = 2
# n_iter = 100
# verbose = F
# a_tau = 0.1
# b_tau = 0.1
mv_lm_uninf_cavi = function(
  Y, X, K = 2, n_iter = 10000, verbose = TRUE
){
  # model dimensions
  M = ncol(Y)
  N = nrow(Y)
  P = ncol(X)

  # run the model
  mvlm_fit = mv_lm_uninf_gibbs_cpp(Y, X, K, n_iter, verbose, a_tau, b_tau)
  final_results = get_params(mvlm_fit, M, P, K)

  return(final_results)
}
