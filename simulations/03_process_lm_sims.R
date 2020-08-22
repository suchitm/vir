#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# process simulation result csv file
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# process lm sim results
lm_results_path = paste0(results_path, "lm_sim_results.csv")
results_df <- read_csv(lm_results_path)

results_summary <-
  results_df %>%
  filter(!is.na(sim_num)) %>%
  group_by(model_type, N) %>%
  summarise(
    mean = mean(mse),
    median = median(mse),
    coverage = mean(coverage),
    ci_length = mean(ci_length),
    mspe = mean(mspe),
    pred_cov = mean(pred_cov)
  ) %>%
  mutate(
    prior = factor(
      str_split(model_type, "_")[[1]][1],
      levels = c("ridge", "lasso", "hs"),
      labels = c("Ridge", "LASSO", "HS")
    ),
    algo = factor(
      str_split(model_type, "_")[[1]][2],
      levels = c("glm", "gibbs", "cavi", "svi"),
      labels = c("GLM", "Gibbs", "CAVI", "SVI")
    ),
    corr = factor(
      str_split(model_type, "_")[[1]][3],
      levels = c("1se", "min", "corr", "indep"),
      labels = c("1SE", "Min", "Corr", "Indep")
    )
  ) %>% # select(model_type, prior, algo, corr) %>% print(n = 100)
  ungroup() %>%
  filter(!is.na(model_type)) %>%
  arrange(N, prior, algo, corr) %>%
  select(N, prior, algo, corr, median, coverage, mspe)

this_N = results_summary$N %>% unique()
this_priors = results_summary$prior %>% unique() %>% as.character()
prior_inds = list(1:7, 8:14, 15:19)
metric_inds = list(1:3, 4:6, 7:9)
metric_names = c("MSE", "Cov.", "MSPE")
res_mat = matrix(nrow = 19, ncol = 9, NA)
row_labels = c()

for(i in 1:length(prior_inds))
{
  prior_name = this_priors[i]
  row_index = prior_inds[[i]]
  for(j in 1:length(metric_inds))
  {
    col_index = metric_inds[[j]]

    these_results <-
      results_summary %>%
      filter(prior == prior_name, N == this_N[j]) %>%
      mutate(
        corr = as.character(corr),
        corr = ifelse(is.na(corr), "Corr", corr),
        comb = paste0(algo, "-", corr)
      )

    these_labels = these_results$comb

    res_mat[row_index, col_index] <-
      these_results %>%
      select(median, coverage, mspe) %>%
      round(3) %>%
      as.matrix()
  }
  row_labels = c(row_labels, these_labels)
}

res_mat <- format_numbers(res_mat, digits = 3)

lm_tab_path = paste0(REPO_PATH, "/paper/tables/lm_sims.tex")
latex(
  res_mat,
  file = lm_tab_path,
  rowlabel = "",
  rowname = row_labels,
  rgroup = this_priors,
  n.rgroup = lapply(prior_inds, length) %>% unlist(),
  colheads = rep(metric_names, length(this_N)),
  cgroup = paste0("N = ", this_N),
  n.cgroup = rep(length(metric_names), length(this_N)),
  table.env = FALSE,
  col.just = rep('c', 9)
)
