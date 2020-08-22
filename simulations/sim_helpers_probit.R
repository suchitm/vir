# simulate data ------------------------------------------------------------
sim_data_binom = function(N, P, rho = 0.5, type)
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
  z = rnorm(1) + X %*% b
  if(type == "probit") {
    probs = pnorm(z)
  } else if (type == "logit") {
    probs = 1 / (1 + exp(-z))
  }
  y = rep(NA, N)
  for(n in 1:N)
  {
    y[n] = rbinom(1, 1, probs[n])
  }

  retl = list(
    X_train = X[train_ind, ],
    y_train = y[train_ind],
    X_test = X[test_ind, ],
    y_test = y[test_ind],
    b = b
  )
  return(retl)
}

# metric functions --------------------------------------------------
get_model_info_probit_vi = function(
  model_fit, true_b, X_test, y_test, X_train, y_train
){
  mse = mean((true_b - model_fit$mu_b)^2)
  conf_ints = model_fit %>% get_ci_vb(P)
  ci_info = get_coverage(conf_ints, true_b)
  coverage = mean(ci_info$covers)
  ci_length = mean(ci_info$ci_length)

  # predicted probabilities
  preds_train <- predict_probit_vi(model_fit, X_train) %>% as.numeric()
  preds_test <- predict_probit_vi(model_fit, X_test) %>% as.numeric()

  # auc roc and pr
  fg <- preds_test[y_test == 1]
  bg <- preds_test[y_test == 0]
  roc <- roc.curve(scores.class0 = fg, scores.class1 = bg)
  auc_roc <- roc$auc
  pr <- pr.curve(scores.class0 = fg, scores.class1 = bg)
  auc_pr <- pr$auc.integral

  # ppv using f1 score
  ppv <- ppv_w_f1(preds_train, y_train, preds_test, y_test)

  # variable selection
  estim_clust = 1 * ((conf_ints[, 1] > 0) | (conf_ints[, 2] < 0))
  true_clust = 1 * (abs(true_b) > 0)
  rand_ind = rand.index(estim_clust, true_clust)

  retl = list(
    mse = mse,
    coverage = coverage,
    ci_length = ci_length,
    auc_roc = auc_roc,
    auc_pr = auc_pr,
    ppv = ppv,
    rand_ind = rand_ind
  )
  return(retl)
}

get_model_info_probit_gibbs = function(
  model_fit, true_b, seq_to_keep, X_test, y_test, X_train, y_train
){
  coefs = colMeans(model_fit$b_mat[seq_to_keep, ])
  mse = mean((true_b - coefs)^2)
  conf_ints = model_fit %>% get_ci_mcmc()
  ci_info = get_coverage(conf_ints, true_b)
  coverage = mean(ci_info$covers)
  ci_length = mean(ci_info$ci_length)

  # predicted probabilities
  preds_train <- predict_probit_gibbs(model_fit, X_train, seq_to_keep)
  preds_test <- predict_probit_gibbs(model_fit, X_test, seq_to_keep)

  # auc roc and pr
  fg <- preds_test[y_test == 1]
  bg <- preds_test[y_test == 0]
  roc <- roc.curve(scores.class0 = fg, scores.class1 = bg)
  auc_roc <- roc$auc
  pr <- pr.curve(scores.class0 = fg, scores.class1 = bg)
  auc_pr <- pr$auc.integral

  # ppv using f1 score
  ppv <- ppv_w_f1(preds_train, y_train, preds_test, y_test)

  # variable selection
  estim_clust = 1 * ((conf_ints[, 1] > 0) | (conf_ints[, 2] < 0))
  true_clust = 1 * (abs(true_b) > 0)
  rand_ind = rand.index(estim_clust, true_clust)

  retl = list(
    mse = mse,
    coverage = coverage,
    ci_length = ci_length,
    auc_roc = auc_roc,
    auc_pr = auc_pr,
    ppv = ppv,
    rand_ind = rand_ind
  )
  return(retl)
}

get_model_info_probit_glmnet = function(
  model_fit, true_b, coef_type, X_test, y_test, X_train, y_train
){
  preds_train = predict(model_fit, X_train, s = coef_type, type = "response")
  preds_test = predict(model_fit, X_test, s = coef_type, type = "response")

  # auc roc and pr
  fg <- preds_test[y_test == 1]
  bg <- preds_test[y_test == 0]
  roc <- roc.curve(scores.class0 = fg, scores.class1 = bg)
  auc_roc <- roc$auc
  pr <- pr.curve(scores.class0 = fg, scores.class1 = bg)
  auc_pr <- pr$auc.integral

  # ppv using f1 score
  ppv <- ppv_w_f1(preds_train, y_train, preds_test, y_test)

  # variable selection
  coefs = coef(model_fit, s = coef_type)[-1]
  estim_clust = 1 * (abs(coefs) > 0)
  true_clust = 1 * (abs(true_b) > 0)
  rand_ind = rand.index(estim_clust, true_clust)

  # mse
  mse = mean((coefs - true_b)^2)

  retl = list(
    mse = mse,
    coverage = NA,
    ci_length = NA,
    auc_roc = auc_roc,
    auc_pr = auc_pr,
    ppv = ppv,
    rand_ind = rand_ind
  )
  return(retl)
}

#-------------------------------------------------------------------------
# models for fit
#-------------------------------------------------------------------------
fit_model_probit = function(
  X_train, y_train, X_test, y_test, model_type, true_b
){
  # gibbs sampling iterations
  gibbs_iter = 5000
  seq_to_keep = 2000:5000
  svi_n_iter = 15000
  cavi_n_iter = 1000

  # fit model, get mse and coverage
  switch(
    model_type,
    # glmnet models
    ridge_glm_1se = {
      model_fit = cv.glmnet(X_train, y_train, alpha = 0, family = "binomial")
      coefs = coef(model_fit, s = 'lambda.1se')
      model_info = get_model_info_probit_glmnet(
        model_fit, true_b, "lambda.1se", X_test, y_test, X_train, y_train
      )
    },
    ridge_glm_min = {
      model_fit = glmnet::cv.glmnet(X_train, y_train, alpha = 0)
      coefs = coef(model_fit, s = 'lambda.min')
      model_info = get_model_info_probit_glmnet(
        model_fit, true_b, "lambda.min", X_test, y_test, X_train, y_train
      )
    },
    lasso_glm_min = {
      model_fit = glmnet::cv.glmnet(X_train, y_train)
      coefs = coef(model_fit, s = 'lambda.min')
      model_info = get_model_info_probit_glmnet(
        model_fit, true_b, "lambda.min", X_test, y_test, X_train, y_train
      )
    },
    lasso_glm_1se = {
      model_fit = glmnet::cv.glmnet(X_train, y_train)
      preds = predict(model_fit, X_test)
      coefs = coef(model_fit, s = 'lambda.1se')
      model_info = get_model_info_probit_glmnet(
        model_fit, true_b, "lambda.1se", X_test, y_test, X_train, y_train
      )
    },
    # gibbs samplers
    ridge_gibbs = {
      model_fit = fastbayes:::probit_ridge_gibbs(
        y_train, X_train, F, gibbs_iter
      )
      model_info = get_model_info_probit_gibbs(
        model_fit, true_b, seq_to_keep, X_test, y_test, X_train, y_train
      )
    },
    lasso_gibbs = {
      model_fit = fastbayes:::probit_lasso_gibbs(
        y_train, X_train, F, gibbs_iter
      )
      model_info = get_model_info_probit_gibbs(
        model_fit, true_b, seq_to_keep, X_test, y_test, X_train, y_train
      )
    },
    hs_gibbs = {
      model_fit = fastbayes:::probit_hs_gibbs(
        y_train, X_train, F, gibbs_iter
      )
      model_info = get_model_info_probit_gibbs(
        model_fit, true_b, seq_to_keep, X_test, y_test, X_train, y_train
      )
    },
    #----------------------------------------#
    # variational bayes
    #----------------------------------------#
    #~~~~~~~~~~~~~~~~~~~~#
    #  ridge
    #~~~~~~~~~~~~~~~~~~~~#
    ridge_cavi_corr = {
      model_fit = fastbayes:::probit_ridge_cavi(
        y_train, X_train, n_iter = cavi_n_iter, verbose = FALSE, type = 0
      )
      model_info = get_model_info_probit_vi(
        model_fit, true_b, X_test, y_test, X_train, y_train
      )
    },
    ridge_cavi_indep = {
      model_fit = fastbayes:::probit_ridge_cavi(
        y_train, X_train, n_iter = cavi_n_iter, verbose = FALSE, type = 1
      )
      model_info = get_model_info_probit_vi(
        model_fit, true_b, X_test, y_test, X_train, y_train
      )
    },
    ridge_svi_corr = {
      model_fit = fastbayes:::probit_ridge_svi(
        y_train, X_train, n_iter = svi_n_iter, verbose = FALSE, type = 0,
        batch_size = 50
      )
      model_info = get_model_info_probit_vi(
        model_fit, true_b, X_test, y_test, X_train, y_train
      )
    },
    ridge_svi_indep = {
      model_fit = fastbayes:::probit_ridge_svi(
        y_train, X_train, n_iter = svi_n_iter, verbose = FALSE, type = 1,
        batch_size = 50
      )
      model_info = get_model_info_probit_vi(
        model_fit, true_b, X_test, y_test, X_train, y_train
      )
    },
    #~~~~~~~~~~~~~~~~~~~~#
    #  lasso
    #~~~~~~~~~~~~~~~~~~~~#
    lasso_cavi_corr = {
      model_fit = fastbayes:::probit_lasso_cavi(
        y_train, X_train, n_iter = cavi_n_iter, verbose = FALSE, type = 0, tol = 0
      )
      model_info = get_model_info_probit_vi(
        model_fit, true_b, X_test, y_test, X_train, y_train
      )
    },
    lasso_cavi_indep = {
      model_fit = fastbayes:::probit_lasso_cavi(
        y_train, X_train, n_iter = cavi_n_iter, verbose = FALSE, type = 1
      )
      model_info = get_model_info_probit_vi(
        model_fit, true_b, X_test, y_test, X_train, y_train
      )
    },
    lasso_svi_corr = {
      model_fit = fastbayes:::probit_lasso_svi(
        y_train, X_train, n_iter = svi_n_iter, verbose = FALSE, type = 0,
        batch_size = 50
      )
      model_info = get_model_info_probit_vi(
        model_fit, true_b, X_test, y_test, X_train, y_train
      )
    },
    lasso_svi_indep = {
      model_fit = fastbayes:::probit_lasso_svi(
        y_train, X_train, n_iter = svi_n_iter, verbose = FALSE, type = 1,
        batch_size = 50
      )
      model_info = get_model_info_probit_vi(
        model_fit, true_b, X_test, y_test, X_train, y_train
      )
    },
    #~~~~~~~~~~~~~~~~~~~~#
    #  hs
    #~~~~~~~~~~~~~~~~~~~~#
    hs_cavi_corr = {
      model_fit = fastbayes:::probit_hs_cavi(
        y_train, X_train, n_iter = cavi_n_iter, verbose = FALSE, type = 0
      )
      model_info = get_model_info_probit_vi(
        model_fit, true_b, X_test, y_test, X_train, y_train
      )
    },
    hs_cavi_indep = {
      model_fit = fastbayes:::probit_hs_cavi(
        y_train, X_train, n_iter = cavi_n_iter, verbose = FALSE, type = 1
      )
      model_info = get_model_info_probit_vi(
        model_fit, true_b, X_test, y_test, X_train, y_train
      )
    },
    hs_svi_corr = {
      model_fit = fastbayes:::probit_hs_svi(
        y_train, X_train, n_iter = svi_n_iter, verbose = FALSE, type = 0,
        batch_size = 50
      )
      model_info = get_model_info_probit_vi(
        model_fit, true_b, X_test, y_test, X_train, y_train
      )
    },
    hs_svi_indep = {
      model_fit = fastbayes:::probit_hs_svi(
        y_train, X_train, n_iter = svi_n_iter, verbose = FALSE, type = 1,
        batch_size = 50
      )
      model_info = get_model_info_probit_vi(
        model_fit, true_b, X_test, y_test, X_train, y_train
      )
    }
  )
  return(model_info)
}
