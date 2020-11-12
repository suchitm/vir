#' generate predictions for a multivariate fit from VI algorithms
#' @param fit model fit
#' @param X_test matrix of predictor values at which to predict the response
#' @param n_samps number of samples to draw from the variational predictive
#'   distribution
#' @return Array of draws from the predictive distribution, along with the means
#'   of the distributions, and their lower (2.5) and upper (97.5) bounds of the
#'   credible intervals.
#' @export
predict_mv_lm_vi = function(fit, X_test, n_samps)
{
  new_N = nrow(X_test)
  one_N = rep(1, new_N)
  P = ncol(X_test)
  M = nrow(fit$theta$mu)
  K = ncol(fit$theta$mu)

  msigma_theta_chol = chol(fit$theta$msigma)
  msigma_B_chol = chol(fit$B$msigma)

  Y_array = array(dim = c(n_samps, new_N, M), NA)

  for(i in 1:n_samps)
  {
    # draw values
    b0 = fit$b0$mu + rnorm(M) * sqrt(fit$b0$vsigma2)
    B = fit$B$mu + matrix(nrow = M, ncol = P, rnorm(M * P)) %*% msigma_B_chol
    theta = fit$theta$mu +
      matrix(nrow = M, ncol = K, rnorm(M * K)) %*% msigma_theta_chol
    tau = rgamma(n = 1, shape = fit$tau$shape, rate = fit$tau$rate)

    # draw sample
    Y_array[i, , ] = one_N %*% t(b0) + X_test %*% t(B) +
      matrix(nrow = new_N, ncol = K, rnorm(new_N * K)) %*% t(theta) +
      1 / sqrt(tau) * matrix(nrow = new_N, ncol = M, rnorm(new_N * M))
  }

  means = apply(Y_array, MARGIN = c(2, 3), FUN = mean)
  lower = apply(
    Y_array, MARGIN = c(2, 3), FUN = function(x) quantile(x, probs = 0.025)
  )
  upper = apply(
    Y_array, MARGIN = c(2, 3), FUN = function(x) quantile(x, probs = 0.975)
  )

  return(
    samples = Y_array,
    mean = means,
    lower_bound = lower,
    upper_bound = upper
  )
}
