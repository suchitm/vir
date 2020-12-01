# simulate data ------------------------------------------------------------
sim_data_lm = function(N, P, rho = 0.5, snr = 1)
{
  train_ind = 1:N
  test_ind = 1:500 + N
  N = N + 500
  b = rnorm(P)
  b[(floor(P * 0.2) + 1):P] = 0.0
  X = matrix(nrow = N, ncol = P)
  X_cov_mat = matrix(nrow = P, ncol = P)
  for(i in 1:P) {
    for(j in 1:P) {
      X_cov_mat[i, j] = rho^(abs(i - j))
    }
  }
  X_cov_mat_chol = t(chol(X_cov_mat))
  X = t(X_cov_mat_chol %*% matrix(nrow = P, ncol = N, rnorm(N * P)))
  y_star = rnorm(1) + X %*% b
  sigma = max(0.01, sd(y_star) / sqrt(snr))
  y = y_star + rnorm(n = N, mean = 0, sd = sigma)
  retl = list(
    X_train = X[train_ind, ],
    y_train = y[train_ind],
    X_test = X[test_ind, ],
    y_test = y[test_ind],
    b = b,
    sigma = sigma
  )
  return(retl)
}

# metric functions --------------------------------------------------
get_model_info_lm_vi = function(model_fit, true_b, X_test, y_test)
{
  P = ncol(X_test)
  coefs = model_fit$mu_b
  mse = mean((true_b - coefs)^2)
  cred_ints = model_fit %>% get_ci_vb(P)
  ci_info = get_coverage(cred_ints, true_b)
  coverage = mean(ci_info$covers)
  ci_length = mean(ci_info$ci_length)

  # coverage and length for non-zero coefficients
  nz_b_ind = !(true_b == 0)
  nz_true_b = true_b[nz_b_ind]
  nz_cred_int = cred_ints[nz_b_ind, ]
  nz_ci_info = get_coverage(nz_cred_int, nz_true_b)
  nz_coverage = mean(nz_ci_info$covers)
  nz_ci_length = mean(nz_ci_info$ci_length)

  # nz_mse
  nz_mse = mean((coefs[nz_b_ind] - nz_true_b)^2)

  # prediction
  preds <- predict_lm_vi(model_fit, X_test)
  mspe <- sqrt(sum((y_test - preds$estimate)^2)) / sqrt(sum(y_test^2))
  pred_cov <- mean((preds$ci[, 1] <= y_test & y_test <= preds$ci[, 2]))

  # variable selection
  estim_clust = 1 * ((cred_ints[, 1] > 0) | (cred_ints[, 2] < 0))
  true_clust = 1 * (abs(true_b) > 0)
  rand_ind = rand.index(estim_clust, true_clust)

  retl = list(
    mse = mse,
    coverage = coverage,
    ci_length = ci_length,
    mspe = mspe,
    pred_cov = pred_cov,
    rand_ind = rand_ind,
    nz_coverage = nz_coverage,
    nz_mse = nz_mse
  )
  return(retl)
}

get_model_info_lm_gibbs = function(
  model_fit, true_b, seq_to_keep, X_test, y_test
){

  coefs = colMeans(model_fit$b_mat[seq_to_keep, ])
  mse = mean((true_b - coefs)^2)
  cred_ints = model_fit %>% get_ci_mcmc()
  ci_info = get_coverage(cred_ints, true_b)
  coverage = mean(ci_info$covers)
  ci_length = mean(ci_info$ci_length)

  # coverage and length for non-zero coefficients
  nz_b_ind = !(true_b == 0)
  nz_true_b = true_b[nz_b_ind]
  nz_cred_int = cred_ints[nz_b_ind, ]
  nz_ci_info = get_coverage(nz_cred_int, nz_true_b)
  nz_coverage = mean(nz_ci_info$covers)
  nz_ci_length = mean(nz_ci_info$ci_length)

  # nz_mse
  nz_mse = mean((coefs[nz_b_ind] - nz_true_b)^2)

  y_pred = matrix(nrow = length(seq_to_keep), ncol = nrow(X_test))
  for(i in 1:length(seq_to_keep))
  {
    mu = model_fit$b0_vec[i] + X_test %*% model_fit$b_mat[i, ]
    sigma = sqrt(1 / (model_fit$tau_vec[i]))
    y_pred[i, ] = mu + rnorm(nrow(X_test)) * sigma
  }

  # predictions
  estims = colMeans(y_pred)
  mspe = sqrt(sum((y_test - estims)^2)) / sqrt(sum(y_test^2))
  ci_mat = summary(mcmc(y_pred), quantiles = c(0.025, 0.975))$quantiles
  pred_cov = get_coverage(ci_mat, y_test)$covers %>% mean()

  # variable selection
  estim_clust = 1 * ((cred_ints[, 1] > 0) | (cred_ints[, 2] < 0))
  true_clust = 1 * (abs(true_b) > 0)
  rand_ind = rand.index(estim_clust, true_clust)

  retl = list(
    mse = mse,
    coverage = coverage,
    ci_length = ci_length,
    mspe = mspe,
    pred_cov = pred_cov,
    rand_ind = rand_ind,
    nz_coverage = nz_coverage,
    nz_mse = nz_mse
  )
  return(retl)
}

get_model_info_lm_rstan = function(model_fit, true_b, X_test, y_test)
{
  # mse
  coefs = coef(model_fit)
  mse = mean((coefs[-1] - true_b)^2)

  cis = posterior_interval(model_fit, prob = 0.95)
  cred_ints = cis[-c(1, nrow(cis)), ]
  ci_info = get_coverage(cred_ints, true_b)
  coverage = mean(ci_info$covers)
  ci_length = mean(ci_info$ci_length)

  # coverage and length for non-zero coefficients
  nz_b_ind = !(true_b == 0)
  nz_true_b = true_b[nz_b_ind]
  nz_cred_int = cred_ints[nz_b_ind, ]
  nz_ci_info = get_coverage(nz_cred_int, nz_true_b)
  nz_coverage = mean(nz_ci_info$covers)
  nz_ci_length = mean(nz_ci_info$ci_length)

  # nz_mse
  nz_mse = mean((coefs[-1][nz_b_ind] - nz_true_b)^2)

  # predictions
  test_df = as_tibble(data.frame(X_test))
  # this function only returns predictions for 100 of the rows
  estims = cbind(1, X_test) %*% coefs
  mspe = sqrt(sum((y_test - estims)^2)) / sqrt(sum(y_test^2))

  # variable selection
  estim_clust = 1 * ((cred_ints[, 1] > 0) | (cred_ints[, 2] < 0))
  true_clust = 1 * (abs(true_b) > 0)
  rand_ind = rand.index(estim_clust, true_clust)

  retl = list(
    mse = mse,
    coverage = coverage,
    ci_length = ci_length,
    mspe = mspe,
    pred_cov = NA,
    rand_ind = rand_ind,
    nz_coverage = nz_coverage,
    nz_mse = nz_mse
  )

  return(retl)
}

get_model_info_lm_glmnet = function(
  model_fit, true_b, coef_type, X_test, y_test, X_train, y_train
){
  # estimation
  coefs = coef(model_fit, s = 'lambda.1se')
  mse = mean((true_b - coefs[-1])^2)

  # coverage and length for non-zero coefficients
  nz_b_ind = !(true_b == 0)
  nz_true_b = true_b[nz_b_ind]
  nz_mse = mean((coefs[-1][nz_b_ind] - nz_true_b)^2)

  # prediction
  preds = predict(model_fit, X_test, s = "lambda.1se")
  mspe = sqrt(sum((preds - y_test)^2)) / sqrt(sum(y_test^2))

  # variable selection
  coefs = coef(model_fit, s = coef_type)[-1]
  estim_clust = 1 * (abs(coefs) > 0)
  true_clust = 1 * (abs(true_b) > 0)
  rand_ind = rand.index(estim_clust, true_clust)

  retl = list(
    mse = mse,
    coverage = NA,
    ci_length = NA,
    mspe = mspe,
    pred_cov = NA,
    rand_ind = rand_ind,
    nz_coverage = NA,
    nz_mse = nz_mse
  )

  return(retl)
}

#-------------------------------------------------------------------------
# modesl for fit
#-------------------------------------------------------------------------
fit_model_lm = function(X_train, y_train, X_test, y_test, model_type, true_b)
{
  # fit model, get mse and coverage
  switch(
    model_type,
    # glmnet models
    ridge_glm_1se = {
      model_fit = glmnet::cv.glmnet(X_train, y_train, alpha = 0)
      model_info = get_model_info_lm_glmnet(
        model_fit, true_b, "lambda.1se", X_test, y_test, X_train, y_train
      )
    },
    ridge_glm_min = {
      model_fit = glmnet::cv.glmnet(X_train, y_train, alpha = 0)
      model_info = get_model_info_lm_glmnet(
        model_fit, true_b, "lambda.min", X_test, y_test, X_train, y_train
      )
    },
    lasso_glm_min = {
      model_fit = glmnet::cv.glmnet(X_train, y_train)
      model_info = get_model_info_lm_glmnet(
        model_fit, true_b, "lambda.min", X_test, y_test, X_train, y_train
      )
    },
    lasso_glm_1se = {
      model_fit = glmnet::cv.glmnet(X_train, y_train)
      model_info = get_model_info_lm_glmnet(
        model_fit, true_b, "lambda.1se", X_test, y_test, X_train, y_train
      )
    },
    # gibbs samplers
    ridge_gibbs = {
      model_fit = fastbayes:::lm_ridge_gibbs(y_train, X_train, F, gibbs_iter)
      model_info = get_model_info_lm_gibbs(
        model_fit, true_b, seq_to_keep, X_test, y_test
      )
    },
    lasso_gibbs = {
      model_fit = fastbayes:::lm_lasso_gibbs(y_train, X_train, FALSE, gibbs_iter)
      model_info = get_model_info_lm_gibbs(
        model_fit, true_b, seq_to_keep, X_test, y_test
      )
    },
    hs_gibbs = {
      model_fit = fastbayes:::lm_hs_gibbs(y_train, X_train, F, gibbs_iter)
      model_info = get_model_info_lm_gibbs(
        model_fit, true_b, seq_to_keep, X_test, y_test
      )
    },
    #----------------------------------------#
    # variational bayes
    #----------------------------------------#
    #~~~~~~~~~~~~~~~~~~~~#
    #  ridge
    #~~~~~~~~~~~~~~~~~~~~#
    ridge_cavi_corr = {
      model_fit = fastbayes:::lm_ridge_cavi(
        y_train, X_train, n_iter = cavi_n_iter, verbose = FALSE, type = 0
      )
      model_info = get_model_info_lm_vi(model_fit, true_b, X_test, y_test)
    },
    ridge_cavi_indep = {
      model_fit = fastbayes:::lm_ridge_cavi(
        y_train, X_train, n_iter = cavi_n_iter, verbose = FALSE, type = 1
      )
      model_info = get_model_info_lm_vi(model_fit, true_b, X_test, y_test)
    },
    ridge_svi_corr = {
      model_fit = fastbayes:::lm_ridge_svi(
        y_train, X_train, n_iter = svi_n_iter, verbose = FALSE, type = 0,
        batch_size = 50
      )
      model_info = get_model_info_lm_vi(model_fit, true_b, X_test, y_test)
    },
    ridge_svi_indep = {
      model_fit = fastbayes:::lm_ridge_svi(
        y_train, X_train, n_iter = svi_n_iter, verbose = FALSE, type = 1,
        batch_size = 50
      )
      model_info = get_model_info_lm_vi(model_fit, true_b, X_test, y_test)
    },
    ridge_stan = {
      df = as_tibble(data.frame(y = y_train, X_train))
      model_fit = stan_glm(
        y ~ ., family = gaussian(), data = df,
        prior_intercept = normal(location = 0, scale = 10^6),
        prior = normal(), algorithm = "meanfield",
        QR = TRUE, iter = stan_iter, tol_rel_obj = stan_rel_tol,
        refresh = 0, importance_resampling = FALSE
      )
      model_info = get_model_info_lm_rstan(model_fit, true_b, X_test, y_test)
    },
    #~~~~~~~~~~~~~~~~~~~~#
    # lasso
    #~~~~~~~~~~~~~~~~~~~~#
    lasso_cavi_corr = {
      model_fit = fastbayes:::lm_lasso_cavi(
        y_train, X_train, verbose = FALSE, n_iter = cavi_n_iter, type = 0
      )
      model_info = get_model_info_lm_vi(model_fit, true_b, X_test, y_test)
    },
    lasso_cavi_indep = {
      model_fit = fastbayes:::lm_lasso_cavi(
        y_train, X_train, verbose = FALSE, n_iter = cavi_n_iter, type = 1
      )
      model_info = get_model_info_lm_vi(model_fit, true_b, X_test, y_test)
    },
    lasso_svi_corr = {
      model_fit = fastbayes:::lm_lasso_svi(
        y_train, X_train, n_iter = svi_n_iter, verbose = FALSE, type = 0,
        batch_size = 50
      )
      model_info = get_model_info_lm_vi(model_fit, true_b, X_test, y_test)
    },
    lasso_svi_indep = {
      model_fit = fastbayes:::lm_lasso_svi(
        y_train, X_train, n_iter = svi_n_iter, verbose = FALSE, type = 1,
        batch_size = 50
      )
      model_info = get_model_info_lm_vi(model_fit, true_b, X_test, y_test)
    },
    lasso_stan = {
      df = as_tibble(data.frame(y = y_train, X_train))
      model_fit = stan_glm(
        y ~ ., family = gaussian(), data = df,
        prior_intercept = normal(location = 0, scale = 10^6),
        prior = laplace(), algorithm = "meanfield",
        QR = FALSE, iter = stan_iter, tol_rel_obj = stan_rel_tol,
        refresh = TRUE, importance_resampling = FALSE
      )
      model_info = get_model_info_lm_rstan(model_fit, true_b, X_test, y_test)
    },
    #~~~~~~~~~~~~~~~~~~~~#
    # hs
    #~~~~~~~~~~~~~~~~~~~~#
    hs_cavi_corr = {
      model_fit = fastbayes:::lm_hs_cavi(
        y_train, X_train, verbose = FALSE, type = 0, n_iter = cavi_n_iter,
        a_tau = 0.1, b_tau = 0.1
      )
      model_info = get_model_info_lm_vi(model_fit, true_b, X_test, y_test)
    },
    hs_cavi_indep = {
      model_fit = fastbayes:::lm_hs_cavi(
        y_train, X_train, verbose = FALSE, type = 1, n_iter = cavi_n_iter,
        a_tau = 0.1, b_tau = 0.1
      )
      model_info = get_model_info_lm_vi(model_fit, true_b, X_test, y_test)
    },
    hs_svi_corr = {
      model_fit = fastbayes:::lm_hs_svi(
        y_train, X_train, n_iter = svi_n_iter, verbose = FALSE, type = 0,
        batch_size = 50
      )
      model_info = get_model_info_lm_vi(model_fit, true_b, X_test, y_test)
    },
    hs_svi_indep = {
      model_fit = fastbayes:::lm_hs_svi(
        y_train, X_train, n_iter = svi_n_iter, verbose = FALSE, type = 1,
        batch_size = 50
      )
      model_info = get_model_info_lm_vi(model_fit, true_b, X_test, y_test)
    },
    hs_stan = {
      df = as_tibble(data.frame(y = y_train, X_train))
      model_fit = stan_glm(
        y ~ ., family = gaussian(), data = df,
        prior_intercept = normal(location = 0, scale = 10^6),
        prior = hs(), algorithm = "meanfield",
        QR = FALSE, iter = stan_iter, tol_rel_obj = stan_rel_tol,
        importance_resampling = FALSE, refresh = 1
      )
      model_info = get_model_info_lm_rstan(model_fit, true_b, X_test, y_test)
    }
  )
  return(model_info)
}

#-------------------------------------------------------------------------
# run the simulations for a set of files and models
#-------------------------------------------------------------------------
run_sims_lm = function(models, lm_data)
{
  results_df = tibble(
    sim_num = NA, model_type = NA, mse = NA, coverage = NA, ci_length = NA,
    mspe = NA, pred_cov = NA, rand_ind = NA, nz_coverage = NA, nz_mse = NA
  )

  iter = 0
  for(file in lm_data)
  {
    iter = iter + 1
    # load data
    fpath = paste0(FPATH, file)
    load(fpath)
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
          pred_cov = model_results$pred_cov,
          rand_ind = model_results$rand_ind,
          nz_coverage = model_results$nz_coverage,
          nz_mse = model_results$nz_mse
        ) %>%
        bind_rows(results_df)
    }
    cat("#--------------------------------------------------#", '\n')
    cat("Done with Sim", iter, "of", length(lm_data), '\n')
    cat("#--------------------------------------------------#", '\n')
  }
  return(results_df)
}
