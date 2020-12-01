probit_results_path = paste0(results_path, "probit_sims_vir_results.csv")

FPATH = paste0(REPO_PATH, "/data/")
all_files = list.files(FPATH)
probit_data = all_files[str_detect(all_files, "probit")]

models = c(
  # ridge
  "ridge_cavi_corr", "ridge_cavi_indep", "ridge_svi_corr", "ridge_svi_indep",
  # lasso
  "lasso_cavi_corr", "lasso_cavi_indep", "lasso_svi_corr", "lasso_svi_indep",
  # hs
  "hs_cavi_corr", "hs_cavi_indep", "hs_svi_corr", "hs_svi_indep"
)

results_df <- run_sims_probit(models, probit_data)
write_csv(results_df, path = probit_results_path)
