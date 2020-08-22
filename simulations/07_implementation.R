library(vir)

set.seed(42)
X = matrix(nrow = 100, ncol = 5, rnorm(5 * 100))
colnames(X) = paste0("X", 1:5)
b = rnorm(5)
y = rnorm(1) + X %*% b + rnorm(100)

ridge_cavi_fit = lm_ridge_cavi(y, X, n_iter = 100, rel_tol = 0.0001)

names(ridge_cavi_fit)
names(ridge_cavi_fit$b)

ridge_cavi_fit$b$dist
ridge_cavi_fit$b$mu
round(ridge_cavi_fit$b$sigma_mat, 4)

summary_vi(ridge_cavi_fit, level = 0.95, coef_names = colnames(X))

X_test = matrix(nrow = 5, ncol = 5, rnorm(25))
predict_lm_vi(ridge_cavi_fit, X_test)

ridge_svi_fit = lm_ridge_svi(
  y, X, n_iter = 1000, verbose = TRUE, batch_size = 10, const_rhot = 0.1
)
