probit_results_path = paste0(results_path, "probit_sims_rstan_results.csv")

stan_iter = 1000
stan_rel_obj = 0.01

FPATH = paste0(REPO_PATH, "/data/")
all_files = list.files(FPATH)
probit_data = all_files[str_detect(all_files, "probit")]

models = c("ridge_stan", "lasso_stan", "hs_stan")

results_df <- run_sims_probit(models, probit_data)
write_csv(results_df, path = probit_results_path)
