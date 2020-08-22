library(vir)

X = matrix(nrow = 1000, ncol = 100)
b = rnorm(100)
y = rnorm(1) + X %*% b + rnorm(1000, sd = 0.2)

cavi_fit = lm_ridge_cavi(y, X, n_iter = 100)
