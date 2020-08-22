library(vir)

# generate data
X = matrix(nrow = 1000, ncol = 100, rnorm(100 * 1000))
b = rnorm(100)
b[21:100] = 0
y = rnorm(1) + X %*% b + rnorm(1000)

ridge_cavi_fit = lm_ridge_cavi(y, X, n_iter = 100)
hs_cavi_fit = lm_hs_cavi(y, X, n_iter = 100)

mean((b - hs_cavi_fit$mu_b)^2)
mean((b - ridge_cavi_fit$mu_b)^2)
