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
sim_dat = sim_mv_lm(20, 4, 3, 2)
X = sim_dat$X
Y = sim_dat$Y
K = 2
n_iter = 100
verbose = F
a_tau = 0.1
b_tau = 0.1
rel_tol = 0.0001

mv_lm_uninf_cavi = function(
  Y, X, K = 2, n_iter = 1000, rel_tol = 0.0001, verbose = TRUE,
  a_tau = 0.1, b_tau = 0.1
){
  # model dimensions
  M = ncol(Y)
  N = nrow(Y)
  P = ncol(X)

  # run the model
  mvlm_fit = mv_lm_uninf_cavi_cpp(Y, X, K, n_iter, verbose, a_tau, b_tau)
  final_results = get_params(mvlm_fit, M, P, K)

  return(final_results)
}

get_params_vi = function(mvlm_fit, M, P, K)
{
  # turn theta var-cov into array
  var_theta_array = array(dim = c(M, K, K), data = NA)
  for(m in 1:M)
  {
    start = (m - 1) * K + 1
    end = start + K - 1
    var_theta_array[m, , ] = mvlm_fit$msigma_mat_theta[start:end, ]
  }

  # turn B var-cov into array
  var_b_array = array(dim = c(M, P, P), data = NA)
  for(m in 1:M)
  {
    start = (m - 1) * P + 1
    end = start + P - 1
    var_b_array[m, , ] = mvlm_fit$msigma_mat_b[start:end, ]
  }

  b0 = vector("list")
  b0$dist = 'multivariate normal - independent'
  b0$mu = mvlm_fit$vmu_b0
  b0$vsigma2 = mvlm_fit$vsigma2_b0


  return(retl)
}
