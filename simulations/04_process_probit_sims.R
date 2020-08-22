#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# process simulation result csv file
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
probit_results_path = paste0(results_path, "probit_sim_results.csv")
results_df <- read_csv(probit_results_path)

results_summary <-
  results_df %>%
  filter(!is.na(sim_num)) %>%
  group_by(model_type, N, sim_type) %>%
  summarise(
    mean = mean(mse),
    median = median(mse),
    coverage = mean(coverage),
    ci_length = mean(ci_length),
    auc_roc = mean(auc_roc),
    auc_pr = mean(auc_pr),
    ppv = mean(ppv),
    rand = mean(rand)
  ) %>%
  mutate(
    prior = factor(
      str_split(model_type, "_")[[1]][1],
      levels = c("ridge", "lasso", "hs"),
      labels = c("Ridge", "LASSO", "HS")
    ),
    corr = factor(
      str_split(model_type, "_")[[1]][3],
      levels = c("1se", "min", "corr", "indep"),
      labels = c("1SE", "Min", "Corr", "Indep")
    ),
    algo = factor(
      str_split(model_type, "_")[[1]][2],
      levels = c("glm", "gibbs", "cavi", "svi"),
      labels = c("GLM", "Gibbs", "CAVI", "SVI")
    ),
  ) %>%
  ungroup() %>%
  filter(!is.na(model_type)) %>%
  arrange(N, prior, algo, corr) %>%
  select(
    N, sim_type, prior, algo, corr, mean, coverage, auc_roc, auc_pr, ppv, rand
  ) %>%
  mutate(
    mean = case_when(
      algo == "GLM" & sim_type == "probit" ~ NA_real_,
      algo != "GLM" & sim_type == "logit" ~ NA_real_,
      TRUE ~ mean
    )
  )

this_N = results_summary$N %>% unique()
this_sim_type = results_summary$sim_type %>% unique()
this_priors = results_summary$prior %>% unique() %>% as.character()
prior_inds = list(1:7, 8:14, 15:19)
metric_names = c("MSE", "AUC-PR", "PPV", "RAND")
metric_inds = list(1:4, 5:8)
res_mat = matrix(nrow = 19, ncol = 8, NA)
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
      filter(prior == prior_name, sim_type == this_sim_type[j]) %>%
      mutate(
        corr = as.character(corr),
        corr = ifelse(is.na(corr), "Corr", corr),
        comb = paste0(algo, "-", corr)
      )

    these_labels = these_results$comb

    res_mat[row_index, col_index] <-
      these_results %>%
      select(mean, auc_pr, ppv, rand) %>%
      round(3) %>%
      as.matrix()
  }
  row_labels = c(row_labels, these_labels)
}

res_mat <- format_numbers(res_mat, digits = 3)

probit_tab_path = paste0(REPO_PATH, "/paper/tables/probit_sims.tex")
latex(
  res_mat,
  file = probit_tab_path,
  rowlabel = "",
  rowname = row_labels,
  rgroup = this_priors,
  n.rgroup = lapply(prior_inds, length) %>% unlist(),
  colheads = rep(metric_names, length(this_sim_type)),
  cgroup = c("Logit", "Probit"),
  n.cgroup = rep(length(metric_names), length(this_sim_type)),
  table.env = FALSE,
  col.just = rep('c', ncol(res_mat))
)
