#' Estimate the parameters in a multivariate linear model with the CAVI
#' algorithm. The model uses a multiplicative gamma process for the factor
#' structure
#' @title Multivariate linear regression with a factor model - CAVI
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
mv_lm_uninf_cavi = function(
  Y, X, K = 2, n_iter = 1000, rel_tol = 0.0001, verbose = TRUE,
  a_tau = 0.1, b_tau = 0.1
){
  # model dimensions
  M = ncol(Y)
  N = nrow(Y)
  P = ncol(X)

  # run the model
  mvlm_fit = mv_lm_uninf_cavi_cpp(
    Y, X, K, n_iter, rel_tol, verbose, a_tau, b_tau
  )
  final_results = get_params_vi(mvlm_fit, M, P, K)

  return(final_results)
}

#' Run a Gibbs sampler for the multivariate probit model. Uses the
#' multiplicative gamma process for the factor structure
#' @title Multivariate linear regression with a factor model
#' @param Y matrix of responses
#' @param X matrix of predictors to control for
#' @param K number of factors in the factor model
#' @param n_iter number of iterations to run the Gibbs sampler
#' @param verbose True or False. Print status of the sampler.
#' @param a_tau prior shape parameter for precisions
#' @param b_tau prior shape parameter for precisions
#' @param batch_size Size of the subsamples used to update the parameters.
#' @param cost_rhot Used to set a constant step size in the gradient descent
#'   algorithm. If this parameter is greater than zero, it overrides the step
#'   size iterations calculated using kappa and omega.
#' @param omega Delay for the stepsize (\eqn{\omega}) for the gradient step.
#'   Interacts with \eqn{\kappa} via the formula \eqn{\rho_{t} = (t +
#'   \omega)^{-\kappa}}. This parameter has to be greater than or equal to zero.
#' @param kappa Forgetting rate for the step size iterations; \eqn{\kappa \in
#'   (0.5, 1)}
#' @export
mv_lm_uninf_svi = function(
  Y, X, K = 2, n_iter = 1000, verbose = TRUE, a_tau = 0.1, b_tau = 0.1,
  batch_size = 42, const_rhot = 0.01, omega = 15.0, kappa = 0.6
){
  # model dimensions
  M = ncol(Y)
  N = nrow(Y)
  P = ncol(X)

  # run the model
  mvlm_fit = mvlm_uninf_svi_cpp(
    Y, X, K, n_iter, verbose, a_tau, b_tau, batch_size, const_rhot, omega,
    kappa
  )
  final_results = get_params_vi(mvlm_fit, M, P, K)

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

  B = vector("list")
  B$dist = 'matrix normal - independent over rows'
  B$mu = mvlm_fit$vmu_b
  B$msigma_array = var_b_array

  theta = vector("list")
  theta$dist = 'matrix normal - independent over rows'
  theta$mu = mvlm_fit$vmu_theta
  theta$msigma_array = var_theta_array

  tau = vector("list")
  tau$dist = 'independent gamma'
  tau$shape = mvlm_fit$param_tau$shape
  tau$rate = mvlm_fit$param_tau$rate

  retl = list(b0 = b0, B = B, theta = theta, tau = tau)

  return(retl)
}
