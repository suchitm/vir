#' @title multivariate probit with factor model - variational
#' @param Y integer matrix for the responses
#' @param X matrix of predictors to control for
#' @param K number of factors in the factor model
#' @param n_iter number of iterations to run the algorithm
#' @param n_samps number of samples to draw from the variational distribution
#' @param rel_tol relative tolerance at which to end the algorithm
#' @param verbose print status of the sampler
#' @return A list with samples from the variational distribution for the
#'   intercept term (\code{b0_mat}), regression coefficients (\code{B_array}),
#'   and upper triangular elements of the correlation matrix
#'   (\code{cor_mat_upper_tri})
#' @export
mv_probit_uninf_cavi = function(
  Y, X, K, n_iter = 100, n_samps = 1000, verbose = TRUE, rel_tol = 0.00001
){
  # problem info
  N = nrow(Y)
  M = ncol(Y)
  P = ncol(X)

  # fit the model
  model_fit = mv_probit_uninf_cavi_cpp(Y, X, K, n_iter, verbose, rel_tol)

  # identify params
  params = get_identified_params_vi = function(model_fit, M, P, K, n_samps)

  return(params)
}

#' @title multivariate probit with factor model - variational
#' @param Y integer matrix for the responses
#' @param X matrix of predictors to control for
#' @param K number of factors in the factor model
#' @param n_iter number of iterations to run the algorithm
#' @param samps number samples from the variational distribution
#' @param verbose print status of the sampler
#' @param batch_size Size of the subsamples used to update the parameters.
#' @param cost_rhot Used to set a constant step size in the gradient descent
#'   algorithm. If this parameter is greater than zero, it overrides the step
#'   size iterations calculated using kappa and omega.
#' @param omega Delay for the stepsize (\eqn{\omega}) for the gradient step.
#'   Interacts with \eqn{\kappa} via the formula \eqn{\rho_{t} = (t +
#'   \omega)^{-\kappa}}. This parameter has to be greater than or equal to zero.
#' @param kappa Forgetting rate for the step size iterations; \eqn{\kappa \in
#'   (0.5, 1)}
#' @return A list with samples from the variational distribution for the
#'   intercept term (\code{b0_mat}), regression coefficients (\code{B_array}),
#'   and upper triangular elements of the correlation matrix
#'   (\code{cor_mat_upper_tri})
#' @export
mv_probit_uninf_svi = function(
  Y, X, K, n_iter = 10000, n_samps = 1000, verbose = TRUE, batch_size = 10,
  const_rhot = 0.001, omega = 15.0, kappa = 0.6
){
  # problem info
  N = nrow(Y)
  M = ncol(Y)
  P = ncol(X)

  # fit the model
  model_fit = mv_probit_uninf_cavi_cpp(Y, X, K, n_iter, verbose, rel_tol)

  # identify params
  params = get_identified_params_vi = function(model_fit, M, P, K, n_samps)

  return(params)
}

get_identified_params_vi = function(model_fit, M, P, K, n_samps)
{
  upper_tri_ind = matrix(nrow = M, ncol = M, NA) %>% upper.tri(., diag = FALSE)
  b0_mat = matrix(nrow = n_samps, ncol = M, NA)
  B_array = array(data = NA, dim = c(n_samps, M, P))
  cor_mat_upper_tri = matrix(nrow = n_samps, ncol = sum(upper_tri_ind), NA)

  theta_chol = chol(model_fit$msigma_theta) %>% t()
  B_chol = chol(model_fit$msigma_B) %>% t()

  for(i in 1:n_samps)
  {
    # ----- simulate from the parameters -----
    # B
    rand_mat = matrix(nrow = P, ncol = M, rnorm(M * P))
    estim_B = model_fit$mu_B + t(B_chol %*% rand_mat)

    # theta
    rand_mat = matrix(nrow = K, ncol = M, rnorm(M * K))
    estim_mtheta = model_fit$mu_theta + t(theta_chol %*% rand_mat)

    # b0
    estim_b0 = model_fit$mu_b0 + rnorm(M) * sqrt(model_fit$vsigma2_b0)

    # cov_mat
    estim_cov_mat = estim_mtheta %*% t(estim_mtheta) + diag(1, M)

    # scale to indentified parameters
    D_m12 = diag(1 / sqrt(diag(estim_cov_mat)))
    estim_cor_mat = D_m12 %*% estim_cov_mat %*% D_m12
    estim_B_star = D_m12 %*% estim_B
    estim_b0_star = as.numeric(D_m12 %*% estim_b0)

    # store estimates
    b0_mat[i, ] = estim_b0_star
    B_array[i, , ]  = estim_B_star
    cor_mat_upper_tri[i, ] = estim_cor_mat[upper_tri_ind]
    if(i %% 10 == 0) {
      cat("Done with sample", i, "of", n_samps, '\r')
    }
  }

  retl = list(
    b0_mat = b0_mat,
    B_array = B_array,
    cor_mat_upper_tri = cor_mat_upper_tri
  )
  return(retl)
}
