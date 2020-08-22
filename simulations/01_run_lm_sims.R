#**********************************************************************#
# LM simulation
#**********************************************************************#
N = c(100, 1000, 5000)
P = 50
rho = 0.5
snr = 1
n_sim = 50

models = c(
  # ridge
  "ridge_glm_min", "ridge_glm_1se", "ridge_gibbs", "ridge_cavi_corr",
  "ridge_svi_corr", "ridge_cavi_indep", "ridge_svi_indep",
  # lasso
  "lasso_glm_min", "lasso_glm_1se", "lasso_gibbs", "lasso_cavi_corr",
  "lasso_svi_corr", "lasso_cavi_indep", "lasso_svi_indep",
  # hs
  "hs_gibbs", "hs_cavi_corr", "hs_svi_corr", "hs_cavi_indep", "hs_svi_indep"
)
model_names = models
results_df = tibble(
  sim_num = NA, model_type = NA, mse = NA, coverage = NA, ci_length = NA,
  mspe = NA, pred_cov = NA
)
sim_settings = expand.grid(N, P, rho, snr)
colnames(sim_settings) = c("N", "P", "rho", "snr")

for(i in 1:n_sim)
{
  for(n in 1:length(N))
  {
    for(r in 1:length(rho))
    {
      this_N = N[n]
      this_rho = rho[r]
      # simulate data
      this_dat = sim_data_lm(this_N, P, this_rho, snr)
      y_train = this_dat$y_train
      X_train = this_dat$X_train
      y_test = this_dat$y_test
      X_test = this_dat$X_test
      true_b = this_dat$b
      # run models and store results
      for(model_type in models)
      {
        cat("Starting Model:", model_type, '\n')
        model_results = fit_model_lm(
          X_train, y_train, X_test, y_test, model_type, true_b
        )
        results_df =
          tibble(
            sim_num = i,
            N = this_N,
            model_type = model_type,
            mse = model_results$mse,
            coverage = model_results$coverage,
            ci_length = model_results$ci_length,
            mspe = model_results$mspe,
            pred_cov = model_results$pred_cov
          ) %>%
          bind_rows(results_df)
      }
    }
  }
  cat("#--------------------------------------------------#", '\n')
  cat("Done with Sim", i, '\n')
  cat("#--------------------------------------------------#", '\n')
}

lm_results_path = paste0(results_path, "lm_sim_results.csv")
write_csv(results_df, path = lm_results_path)

