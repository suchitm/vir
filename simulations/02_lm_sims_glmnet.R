lm_results_path = paste0(results_path, "lm_sims_glmnet_results.csv")

FPATH = paste0(REPO_PATH, "/data/")
all_files = list.files(FPATH)
lm_data = all_files[str_detect(all_files, "lm")]

models = c("ridge_glm_min", "ridge_glm_1se", "lasso_glm_min", "lasso_glm_1se")

results_df <- run_sims_lm(models, lm_data)
write_csv(results_df, file = lm_results_path)
