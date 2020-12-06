#' @title Predictions from Normal VI regression model fits
#' @param fit the model fit
#' @param X_test matrix of predictor values for which you want to predict the
#'   response
#' @export
predict_lm_vi = function(fit, X_test)
{
  N = nrow(X_test)
  mu_tau = fit$tau$shape / fit$tau$rate
  mu = fit$b0$mu + X_test %*% fit$b$mu
  sigma2 = diag(fit$b0$var, N) + X_test %*% fit$b$sigma_mat %*% t(X_test) +
    diag(1 / mu_tau, N)
  lower = mu - 1.96 * sqrt(diag(sigma2))
  upper = mu + 1.96 * sqrt(diag(sigma2))
  retl <- list(
    estimate = as.numeric(mu),
    ci = cbind(lower, upper)
  )
  return(retl)
}

#' @title Predictions from normal linear regression model - Gibbs
#' @param fit the model fit
#' @param X_test matrix of predictor values for which you want to predict the
#'   response
#' @export
predict_lm_gibbs = function(model_fit, X_test, seq_to_keep)
{
  N = nrow(X_test)
  y_pred = matrix(nrow = length(seq_to_keep), ncol = N)
  
  for(i in 1:length(seq_to_keep))
  {
    mu = model_fit$b0_vec[i] + X_test %*% model_fit$b_mat[i, ]
    sigma = sqrt(1 / (model_fit$tau_vec[i]))
    y_pred[i, ] = mu + rnorm(nrow(X_test)) * sigma
  }
    
  estimate = as.numeric(colMeans(y_pred))
  lower = apply(y_pred, MARGIN = 2, FUN = function(x) quantile(x, probs = 0.025))
  upper = apply(y_pred, MARGIN = 2, FUN = function(x) quantile(x, probs = 0.975))

  retl = list(
    estimate = estimate, 
    lower = lower, 
    upper = upper,
    samples = y_pred
  )
}

#' @title Predictions from probit VI regression model fits
#' @param fit the model fit
#' @param X_test matrix of predictor values for which you want to predict the
#'   response
#' @export
predict_probit_vi = function(model_fit, X_test)
{
  N = nrow(X_test)
  mu = model_fit$b0$mu + X_test %*% model_fit$b$mu
  sigma2 = diag(model_fit$b0$var, N) +
    X_test %*% model_fit$b$sigma_mat %*% t(X_test) + diag(1, N)
  probs = pnorm(mu / sqrt(diag(sigma2)))
  return(probs)
}

#' @title Predictions from probit Gibbs regression model fits
#' @param fit the model fit
#' @param X_test matrix of predictor values for which you want to predict the
#'   response
#' @export
predict_probit_gibbs = function(model_fit, X_test, seq_to_keep)
{
  N = nrow(X_test)
  y_pred = matrix(nrow = length(seq_to_keep), ncol = N)

  for(i in 1:length(seq_to_keep))
  {
    index = seq_to_keep[i]
    mu = model_fit$b0_vec[index] + X_test %*% model_fit$b_mat[index, ]
    z = mu + rnorm(N)
    y_pred[i, ] = 1 * (z > 0)
  }
  probs <- colMeans(y_pred)
  return(probs)
}
