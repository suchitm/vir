library(testthat)
library(vir)

# Generate Test Data
set.seed(0)
N <- 1000
P <- 5
X <- matrix(nrow = N, ncol = P, rnorm(N * P))
colnames(X) <- paste0("X", 1:5)
b <- rnorm(P)
mu <- rnorm(1)

y <- mu + X %*% b + rnorm(N, sd = 0.1)

test_that("Univariate linear regression with normal prior using CAVI", {
  ridge_cavi_fit <- lm_ridge_cavi(y, X, n_iter = 1000, rel_tol = 0.0001)

  expect_equal(ridge_cavi_fit$b0$mu,
    mu,
    tolerance = 1e-2
  )

  # Test b
  expect_equal(
    ridge_cavi_fit$b0$mu,
    mu,
    tolerance = 1e-2
  )

  # Test mu
  expect_equal(
    ridge_cavi_fit$b$mu,
    b,
    tolerance = 1e-2
  )
})
