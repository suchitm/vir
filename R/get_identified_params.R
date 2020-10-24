#' @title Extract identified parameters from MV probit
#' @name get_identified_params
#' @param mv_probit_fit the fit from the multivariate probit model
#' @param M dimension of the covariance matrix; number of responses.
#' @param P number of covariates; excluding intercept
#' @param K number of factors used
#' @param n_iter number of iterations the sampler was run for
#' @param burn_in number of iterations to drop from the sampler run
#' @export
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

#' @title Extract identified parameters from MV probit svi fit
#' @name get_identified_params
#' @param model_fit the fit from the multivariate probit model
#' @param M dimension of the covariance matrix; number of responses.
#' @param P number of covariates; excluding intercept
#' @param K number of factors used
#' @param n_samps number of samples to draw
#' @export
get_identified_params_vi = function(
  model_fit, M, P, K, n_samps, X_test = NULL, Y_test = NULL
){
  upper_tri_ind = matrix(nrow = M, ncol = M, NA) %>% upper.tri(., diag = FALSE)
  b0_mat = matrix(nrow = n_samps, ncol = M, NA)
  B_array = array(data = NA, dim = c(n_samps, M, P))
  cor_mat_upper_tri = matrix(nrow = n_samps, ncol = sum(upper_tri_ind), NA)

  theta_chol = chol(model_fit$msigma_theta) %>% t()
  B_chol = chol(model_fit$msigma_B) %>% t()

  if(!is.null(X_test))
  {
    N = nrow(X_test)
    Y_array = array(dim = c(n_samps, N, M), NA)
    log_pred = rep(NA, n_samps)
  }

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

    if(!is.null(X_test))
    {
      sim_psi = matrix(nrow = N, ncol = K, rnorm(N * K))
      mu_Z = rep(1, N) %*% t(estim_b0) + X_test %*% t(estim_B) +
        sim_psi %*% t(estim_mtheta)
      Z = mu_Z + matrix(nrow = N, ncol = M, rnorm(M * N))
      Y_array[i, , ] = ifelse(Z > 0, 1, 0)
    }

    log_cdf1 = pnorm(mu_Z, log.p = TRUE)
    log_cdf0 = pnorm(-mu_Z, log.p = TRUE)
    log_pred[i] = sum(Y_test * log_cdf1 + (1 - Y_test) * log_cdf0)

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
    cor_mat_upper_tri = cor_mat_upper_tri,
    Y_test_array = Y_array,
    log_pred = log_pred
  )
  return(retl)
}
