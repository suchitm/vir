#' @title Credible intervals from VI model fits
#' @param fit the model fit
#' @param level level of the credible interval. Default is 95\%.
#' @param coef_names vector of names for the predictors. Use `colnames(X)` to
#'   generate.
#' @export
summary_vi = function(fit, level = 0.95, coef_names = NULL)
{
  ret_mat <- matrix(nrow = length(coef_names) + 1, ncol = 3)
  rownames(ret_mat) = c("Intercept", coef_names)

  cutoff = qnorm(level + (1 - level) / 2)

  lower_b0 = fit$b0$mu - cutoff * sqrt(fit$b0$var)
  upper_b0 = fit$b0$mu + cutoff * sqrt(fit$b0$var)

  lower_b = fit$b$mu - cutoff * sqrt(diag(fit$b$sigma_mat))
  upper_b = fit$b$mu + cutoff * sqrt(diag(fit$b$sigma_mat))

  ret_mat[, 1] = c(fit$b0$mu, fit$b$mu)
  ret_mat[, 2] = c(lower_b0, lower_b)
  ret_mat[, 3] = c(upper_b0, upper_b)

  colnames(ret_mat) = c("Estimate", "Lower", "Upper")

  return(ret_mat)
}

#' @title Credible intervals from Gibbs samples - univariate models
#' @param fit the model fit
#' @param seq_to_keep the sequence of iterations to keep in final calculations. 
#' @param level level of the credible interval. Default is 95\%.
#' @param coef_names vector of names for the predictors. Use `colnames(X)` to
#'   generate.
#' @export
summary_gibbs = function(fit, seq_to_keep, level = 0.95, coef_names = NULL)
{
  ret_mat <- matrix(nrow = length(coef_names) + 1, ncol = 3)
  rownames(ret_mat) = c("Intercept", coef_names)

  lc = (1 - level) / 2
  uc = 1 - lc

  estim_b0 = mean(fit$b0_vec[seq_to_keep])
  estim_b = colMeans(fit$b_mat[seq_to_keep, ])

  lower_b0 = quantile(fit$b0_vec[seq_to_keep], probs = lc)
  lower_b0 = quantile(fit$b0_vec[seq_to_keep], probs = uc)

  lower_b = apply(
    fit$b_mat[seq_to_keep, ], MARGIN = 2, 
    FUN = function(x) quantile(x, probs = lc)
  )
  upper_b = apply(
    fit$b_mat[seq_to_keep, ], MARGIN = 2, 
    FUN = function(x) quantile(x, probs = uc)
  )

  ret_mat[, 1] = c(estim_b0, estim_b)
  ret_mat[, 2] = c(lower_b0, lower_b)
  ret_mat[, 3] = c(upper_b0, upper_b)

  colnames(ret_mat) = c("Estimate", "Lower", "Upper")

  return(ret_mat)
}


