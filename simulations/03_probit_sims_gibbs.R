probit_results_path = paste0(results_path, "probit_sims_gibbs_results.csv")

FPATH = paste0(REPO_PATH, "/data/")
all_files = list.files(FPATH)
probit_data = all_files[str_detect(all_files, "probit")]

models = c("ridge_gibbs", "lasso_gibbs", "hs_gibbs")

results_df <- run_sims_probit(models, probit_data)
write_csv(results_df, path = probit_results_path)
