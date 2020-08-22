#------------------------------------------------------------------------#
# confidence intervals for Gibbs samplers and Lasso
#------------------------------------------------------------------------#
get_ci_vb = function(vb_res, P)
{
    ci_mat = matrix(nrow = P, ncol = 2, NA)
    mu_b = vb_res$mu_b
    vsigma2_b = diag(vb_res$msigma_b)
    for(p in 1:P) {
        ci_mat[p, ] = mu_b[p] + c(-1, 1) * 1.96 * sqrt(vsigma2_b[p])
    }
    return(ci_mat)
}

get_ci_mcmc = function(gibbs_res)
{
    b_mcmc = mcmc(gibbs_res$b_mat)
    ci_mat = summary(b_mcmc, quantiles = c(0.025, 0.975))$quantiles
    return(ci_mat)
}

get_coverage = function(ci_mat, b)
{
    P = length(b)
    covers = rep(NA, length(b))
    ci_length = rep(NA, length(b))
    for(p in 1:P)
    {
        ci_length[p] = ci_mat[p, 2] - ci_mat[p, 1]
        covers[p] = (ci_mat[p, 1] <= b[p]) & (b[p] <= ci_mat[p, 2])
    }
    retl = list(
        covers = covers,
        ci_length = ci_length
    )
    return(retl)
}

#--------------------------------------------------#
# other helpers
#--------------------------------------------------#
format_numbers = function(x, digits = 4)
{
  return(ifelse(is.na(x), "-", format(round(x, digits), nsmall = digits)))
}

# metric helpers
ppv_w_f1 <- function(preds_train, y_train, preds_test, y_test)
{
  seq_probs <- seq(0, 1, length = 1000)
  F1 <- rep(NA, 1000)
  for(i in 1:1000)
  {
    pred_pos <- 1 * (preds_train > seq_probs[i])
    TP <- sum((pred_pos == 1) & (y_train == 1))
    FP <- sum((pred_pos == 1) & (y_train == 0))
    FN <- sum((pred_pos == 0) & (y_train == 1))
    F1[i] <- TP / (TP + 1/2 * (FP + FN))
  }
  optim_cutoff <- seq_probs[which.max(F1)]
  pred_pos <- 1 * (preds_test > optim_cutoff)
  TP <- sum((pred_pos == 1) & (y_test == 1))
  FP <- sum((pred_pos == 1) & (y_test == 0))
  FN <- sum((pred_pos == 0) & (y_test == 1))
  ppv <- TP / (TP + FP)
  return(ppv)
}

# #--------------------------------------------------#
# # run one sim
# #--------------------------------------------------#
# run_one_sim = function(sim_settings, models, results_df, sim_num)
# {
#     # simulate data
#     this_N = sim_settings$N
#     this_P = sim_settings$P
#     this_rho = sim_settings$rho
#     this_snr = sim_settings$snr
#     this_dat = sim_data_probit(this_N, P, this_rho)
#     y = this_dat$y
#     X = this_dat$X
#     true_b = this_dat$b
#     # run models and store results
#     for(model_type in models)
#     {
#         cat("Starting Model:", model_type, '\n')
#         model_results = fit_model(X, y, model_type, true_b)
#         results_df =
#             tibble(
#                 sim_num = sim_num,
#                 model_type = model_type,
#                 N = this_N,
#                 P = this_P,
#                 rho = this_rho,
#                 snr = this_snr,
#                 mse = model_results$mse,
#                 coverage = model_results$coverage,
#                 ci_length = model_results$ci_length
#             ) %>%
#             bind_rows(results_df)
#     }
#     return(results_df %>% filter(!is.na(model_type)))
# }
#
# #**********************************************************************#
# # probit helpers
# #**********************************************************************#
# sim_mv_probit = function(N, M, P, K)
# {
#   X = matrix(nrow = N, ncol = P, rnorm(N * P))
#   one_N = rep(1, N)
#   B = matrix(nrow = M, ncol = P, rnorm(M * P))
#   b0 = rnorm(M)
#   mphi = matrix(nrow = N, ncol = K, rnorm(N * K) * 0.5)
#   mtheta = matrix(nrow = M, ncol = K, rnorm(M * K) * 0.5)
#   tau = rgamma(M, 2.1, 1)
#
#   B = matrix(nrow = M, ncol = P, 0.0)
#   b0 = rep(0.0, M)
#   tau = rep(1, M)
#
#   cov_mat = mtheta %*% t(mtheta) + diag(1 / tau)
#   mu_Z = one_N %*% t(b0) + X %*% t(B) + mphi %*% t(mtheta)
#   Z = mu_Z + matrix(nrow = N, ncol = M, rnorm(N * M)) %*% diag(1 / sqrt(tau))
#
#   probs = pnorm(Z)
#   Y = matrix(nrow = N, ncol = M, NA)
#   for(n in 1:N) {
#     for(m in 1:M) {
#       Y[n, m] = rbinom(n = 1, size = 1, prob = probs[n, m])
#     }
#   }
#
#   D_m12 = diag(1 / sqrt(diag(cov_mat)))
#   Z = Z %*% D_m12
#   B = D_m12 %*% B
#   b0 = D_m12 %*% b0
#   cor_mat = D_m12 %*% cov_mat %*% D_m12
#
#   list(
#     X = X, Y = Y, mu_Z = mu_Z, b0 = b0, B = B, cor_mat = cor_mat,
#     mtheta = mtheta, mphi = mphi, tau = tau
#   ) %>% return()
# }
#
# mv_probit_gibbs_estims = function(Y, X, K, n_iter)
# {
#   N = nrow(Y)
#   M = ncol(Y)
#   P = ncol(X)
#
#   seq_to_keep = (n_iter / 2):n_iter
#   fit = fastbayes:::mv_probit_uninf_gibbs(Y, X, K, n_iter = n_iter)
#
#   estim_B = fit$B_mat[seq_to_keep, ] %>%
#     colMeans() %>%
#     matrix(nrow = M, ncol = P, .)
#
#   estim_cor_mat = fit$cov_mat_mat[seq_to_keep, ] %>%
#     colMeans() %>%
#     matrix(nrow = M, ncol = M, .)
#
#   return(list(B = estim_B, cor_mat = estim_cor_mat, fit = fit))
# }
