lm_results_path = paste0(results_path, "lm_sims_gibbs_results.csv")

FPATH = paste0(REPO_PATH, "/data/")
all_files = list.files(FPATH)
lm_data = all_files[str_detect(all_files, "lm")]

models = c("ridge_gibbs", "lasso_gibbs", "hs_gibbs")

results_df <- run_sims_lm(models, lm_data)
write_csv(results_df, path = lm_results_path)
