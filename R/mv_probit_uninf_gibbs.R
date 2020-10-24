#' Run a Gibbs sampler for the multivariate probit model
#' @title Multivariate probit with a factor model
#' @param Y integer matrix for the responses
#' @param X matrix of predictors to control for
#' @param K number of factors in the factor model
#' @param n_iter number of iterations to run the Gibbs sampler
#' @param verbose True or False. Print status of the sampler.
#' @return A list with samples for the intercept term (\code{b0_mat}),
#'   regression coefficients (\code{B_array}), and upper triangular elements of
#'   the correlation matrix (\code{cor_mat_upper_tri})
#' @export
mv_probit_uninf_gibbs = function(
  Y, X, K = 2, n_iter = 10000, burn_in = 5000, verbose = TRUE
){
  # model dimensions
  M = ncol(Y)
  N = nrow(Y)
  P = ncol(X)

  # run the model
  model_fit = mv_probit_uninf_gibbs(Y, X, K, n_iter, burn_in, verbose)

  # rescale the parameters to the
  final_results = get_identified_params(mv_probit_fit, M, P, K)

  return(final_results)
}

get_identified_params = function(mv_probit_fit, M, P, K)
{
  n_iter = nrow(mv_probit_fit$B_mat)

  upper_tri_ind = matrix(nrow = M, ncol = M, NA) %>% upper.tri(., diag = FALSE)
  b0_mat = matrix(nrow = n_iter, ncol = M, NA)
  B_array = array(data = NA, dim = c(n_iter, M, P))
  cor_mat_upper_tri = matrix(nrow = n_iter, ncol = sum(upper_tri_ind), NA)

  for(i in 1:n_iter)
  {
    # get non-identified estimates
    estim_B = mv_probit_fit$B_mat[i, ] %>%
      matrix(nrow = M, ncol = P, .)
    estim_mtheta = mv_probit_fit$mtheta_mat[i, ] %>%
      matrix(nrow = M, ncol = K, .)
    estim_b0 = mv_probit_fit$b0_mat[i, ]
    estim_var = 1.0
    estim_cov_mat = estim_mtheta %*% t(estim_mtheta) + diag(estim_var, M)

    # scale to indentified parameters
    D_m12 = diag(1 / sqrt(diag(estim_cov_mat)))
    estim_cor_mat = D_m12 %*% estim_cov_mat %*% D_m12
    estim_B_star = D_m12 %*% estim_B
    estim_b0_star = as.numeric(D_m12 %*% estim_b0)
    # estim_mtheta_star = D_m12 %*% estim_mtheta

    # store estimates
    b0_mat[i, ] = estim_b0_star
    B_array[i, , ]  = estim_B_star
    cor_mat_upper_tri[i, ] = estim_cor_mat[upper_tri_ind]
  }

  retl = list(
    b0_mat = b0_mat,
    B_array = B_array,
    cor_mat_upper_tri = cor_mat_upper_tri
  )
  return(retl)
}
