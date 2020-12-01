#**************************************************#
# timing experiments
#**************************************************#
N_fix = 1000
P = c(100, 200, 400, 600, 800)
P_fix = 100
N = c(1000, 5000, 10000, 20000, 40000)
NP_comb =
  expand.grid(N_fix, P, "N_fix") %>%
  bind_rows(expand.grid(N, P_fix, "P_fix"))
rho = 0.5
snr = 1
n_sim = 3

results_df <- tibble(model = NA, algo = NA, N = NA, P = NA, time = NA, fix = NA)

for(i in 1:n_sim)
{
  for(j in 1:nrow(NP_comb))
  {
    #---------------------#
    # LM Sims
    #---------------------#
    n = NP_comb[j, 1]
    p = NP_comb[j, 2]
    fix = NP_comb[j, 3]
    this_dat = sim_data_lm(n, p)
    y_train = this_dat$y_train
    X_train = this_dat$X_train
    rm(this_dat)

    #----- GLMNET -----#
    ptm = proc.time()
    fit = cv.glmnet(X_train, y_train)
    time = (proc.time() - ptm)[3]

    results_df <-
      tibble(
        model = "linear", algo = "glmnet", N = n, P = p, time = time, fix = fix
      ) %>%
      bind_rows(results_df)

    #----- CAVI -----#
    ptm = proc.time()
    fit = fastbayes:::lm_hs_cavi(
      y_train, X_train, n_iter = 1000, tol = 0.0001, type = 0
    )
    time = (proc.time() - ptm)[3]

    results_df <-
      tibble(
        model = "linear", algo = "cavi_0", N = n, P = p, time = time, fix = fix
      ) %>%
      bind_rows(results_df)

    ptm = proc.time()
    fit = fastbayes:::lm_hs_cavi(
      y_train, X_train, n_iter = 1000, tol = 0.0001, type = 1
    )
    time = (proc.time() - ptm)[3]

    results_df <-
      tibble(
        model = "linear", algo = "cavi_1", N = n, P = p, time = time, fix = fix
      ) %>%
      bind_rows(results_df)

    #----- SVI -----#
    ptm = proc.time()
    fit = fastbayes:::lm_hs_svi(
      y_train, X_train, n_iter = 15000, batch_size = 100, type = 0
    )
    time = (proc.time() - ptm)[3]

    results_df <-
      tibble(
        model = "linear", algo = "svi_0", N = n, P = p, time = time, fix = fix
      ) %>%
      bind_rows(results_df)

    ptm = proc.time()
    fit = fastbayes:::lm_hs_svi(
      y_train, X_train, n_iter = 15000, batch_size = 100, type = 1
    )
    time = (proc.time() - ptm)[3]

    results_df <-
      tibble(
        model = "linear", algo = "svi_1", N = n, P = p, time = time, fix = fix
      ) %>%
      bind_rows(results_df)

    #---------------------#
    # Probit Sims
    #---------------------#
    this_dat = sim_data_binom(n, p, type = "probit")
    y_train = this_dat$y_train
    X_train = this_dat$X_train
    rm(this_dat)

    #----- GLMNET -----#
    ptm = proc.time()
    fit = cv.glmnet(X_train, y_train, family = "binomial")
    time = (proc.time() - ptm)[3]
    rm(fit)

    results_df <-
      tibble(
        model = "binom", algo = "glmnet", N = n, P = p, time = time, fix = fix
      ) %>%
      bind_rows(results_df)

    #----- CAVI -----#
    ptm = proc.time()
    fit = fastbayes:::probit_hs_cavi(
      y_train, X_train, n_iter = 1000, tol = 0.0001, type = 0
    )
    time = (proc.time() - ptm)[3]
    rm(fit)

    results_df <-
      tibble(
        model = "binom", algo = "cavi_0", N = n, P = p, time = time, fix = fix
      ) %>%
      bind_rows(results_df)

    ptm = proc.time()
    fit = fastbayes:::probit_hs_cavi(
      y_train, X_train, n_iter = 1000, tol = 0.0001, type = 1
    )
    time = (proc.time() - ptm)[3]
    rm(fit)

    results_df <-
      tibble(
        model = "binom", algo = "cavi_1", N = n, P = p, time = time, fix = fix
      ) %>%
      bind_rows(results_df)

    #----- SVI -----#
    ptm = proc.time()
    fit = fastbayes:::probit_hs_svi(
      y_train, X_train, n_iter = 15000, batch_size = 100, type = 0
    )
    time = (proc.time() - ptm)[3]
    rm(fit)

    results_df <-
      tibble(
        model = "binom", algo = "svi_0", N = n, P = p, time = time, fix = fix
      ) %>%
      bind_rows(results_df)

    ptm = proc.time()
    fit = fastbayes:::probit_hs_svi(
      y_train, X_train, n_iter = 15000, batch_size = 100, type = 1
    )
    time = (proc.time() - ptm)[3]
    rm(fit)

    results_df <-
      tibble(
        model = "binom", algo = "svi_1", N = n, P = p, time = time, fix = fix
      ) %>%
      bind_rows(results_df)
  }
  cat("#--------------------------------------------------#", '\n')
  cat("Done with Sim", i, '\n')
  cat("#--------------------------------------------------#", '\n')
}

timing_results_path = paste0(results_path, "timing_results.csv")
write_csv(results_df %>% na.omit(), path = timing_results_path)
