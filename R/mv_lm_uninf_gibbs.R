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
mv_lm_uninf_gibbs = function(
  Y, X, K = 2, n_iter = 10000, burn_in = 5000, verbose = TRUE
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

#' predict from a multivariate linear model
#' @param mvlm_fit fit from a mv_lm
#' @param X_test
#' @return an array with predictions (i, N, M)
#' @export
predict_mv_lm = function(mvlm_fit, X_test)
{
  N_test = nrow(X_test)
  M = nrow(mvlm_fit$B_array[1, , ])
  K = ncol(mvlm_fit$mtheta_array[1, , ])
  n_iter = nrow(mvlm_fit$b0_mat)
  Y_test = array(data = NA, dim = c(n_iter, N_test, M))

  for(i in 1:n_iter)
  {
    mpsi = matrix(nrow = N_test, ncol = K, rnorm(N_test * K))
    randn = matrix(nrow = N_test, ncol = M, rnorm(N_test * M))

    Y_test[i, , ] = X_test %*% t(mvlm_fit$B_array[i, , ]) +
      mpsi %*% t(mvlm_fit$mtheta_array[i, , ]) +
      t(diag(sqrt(mvlm_fit$tau_mat[i, ])) %*% t(randn))

    cat('Done with iteration', i, 'of', n_iter, '\r')
  }

  return(Y_test)
}

sim_mv_lm = function(N, M, P, K)
{
  X = matrix(nrow = N, ncol = P, rnorm(N * P))
  B = matrix(nrow = M, ncol = P, rnorm(M * P))
  mtheta = matrix(nrow = M, ncol = K, rnorm(M * K))
  mpsi = matrix(nrow = N, ncol = K, rnorm(N * K))

  Y = X %*% t(B) + mpsi %*% t(mtheta) + matrix(nrow = N, ncol = M, rnorm(N * M))

  retl = list(X = X, Y = Y)
  return(retl)
}

get_params = function(mvlm_fit, M, P, K)
{
  n_iter = nrow(mvlm_fit$B_mat)

  b0_mat = matrix(nrow = n_iter, ncol = M, NA)
  B_array = array(data = NA, dim = c(n_iter, M, P))
  mtheta_array = array(data = NA, dim = c(n_iter, M, K))
  tau_mat = matrix(nrow = n_iter, ncol = M, NA)

  for(i in 1:n_iter)
  {
    # get non-identified estimates
    B_array[i, , ] = mvlm_fit$B_mat[i, ] %>% matrix(nrow = M, ncol = P, .)
    mtheta_array[i, , ] = mvlm_fit$mtheta_mat[i, ] %>%
      matrix(nrow = M, ncol = K, .)
    cat('Done with mat to array', i, 'of', n_iter, '\r')
  }

  retl = list(
    b0_mat = mvlm_fit$b0_mat,
    B_array = B_array,
    mtheta_array = mtheta_array,
    tau_mat = mvlm_fit$tau_mat,
    loglik_vec = mvlm_fit$loglik_vec
  )
  return(retl)
}
