lm_results_path = paste0(results_path, "lm_sims_vir_results.csv")

svi_n_iter = 100
cavi_n_iter = 500

FPATH = paste0(REPO_PATH, "/data/")
all_files = list.files(FPATH)
lm_data = all_files[str_detect(all_files, "lm")]

models = c(
  # ridge
  "ridge_cavi_corr", "ridge_cavi_indep", "ridge_svi_corr", "ridge_svi_indep",
  # lasso
  "lasso_cavi_corr", "lasso_cavi_indep", "lasso_svi_corr", "lasso_svi_indep",
  # hs
  "hs_cavi_corr", "hs_cavi_indep", "hs_svi_corr", "hs_svi_indep"
)

results_df <- run_sims_lm(models, lm_data)
write_csv(results_df, path = lm_results_path)
