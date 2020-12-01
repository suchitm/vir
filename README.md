
<!-- README.md is generated from README.Rmd. Please edit that file -->

# vir

## Overview

This package includes a set of variational and stochastic variational
algorithms to estimate parameters in linear and probit regression with
shrinkage priors. The algorithms are implemented using RcppEigen and
don’t require linking with any other external libraries.

## Installation

Install the package using `devtools`

``` r
devtools::install_github("suchitm/vir")
```

## Usage

The functions in the package are named according to the link function,
`lm` for normal linear models and `probit` for binary regression,
followed by the shrinkage prior type: `ridge`, `lasso`, `hs`, `uninf`;
and then the algorithm: `gibbs`, `cavi`, or `svi`. Therefore, if an
analyst wishes to use the svi algorithm with a linear model and
horseshoe prior, they can call the function `lm_hs_svi` to analyze the
data.

``` r
library(vir)
set.seed(42)
X = matrix(nrow = 100, ncol = 5, rnorm(5 * 100))
colnames(X) = paste0('X', 1:5)
b = rnorm(5)
y = rnorm(1) + X %*% b + rnorm(100)
ridge_cavi_fit = lm_ridge_cavi(y, X, verbose = F, n_iter = 100, rel_tol = 0.0001)
```

The function outputs a nested list containing the names of the
parameters and their optimal variational distributions.

``` r
names(ridge_cavi_fit)
```

    ## [1] "b0"     "b"      "tau"    "lambda" "elbo"

``` r
names(ridge_cavi_fit$b)
```

    ## [1] "dist"      "mu"        "sigma_mat"

``` r
ridge_cavi_fit$b$dist
```

    ## [1] "multivariate normal"

``` r
ridge_cavi_fit$b$mu
```

    ## [1]  0.87668826  0.92639922  0.08866204  0.10254637 -0.75907903

The parameters can be summarised by:

``` r
summary_vi(ridge_cavi_fit, level = 0.95, coef_names = colnames(X))
```

    ##              Estimate      Lower       Upper
    ## Intercept -0.17491248 -0.3901667  0.04034176
    ## X1         0.87668826  0.6711879  1.08218858
    ## X2         0.92639922  0.6909048  1.16189365
    ## X3         0.08866204 -0.1218641  0.29918818
    ## X4         0.10254637 -0.1424468  0.34753954
    ## X5        -0.75907903 -0.9717937 -0.54636432

### Prediction

Using the model fit, predictions can be generated using the predict
functions named after the link and the algorithm (`vi` or `gibbs`)

``` r
X_test = matrix(nrow = 5, ncol = 5, rnorm(25))
predict_lm_vi(ridge_cavi_fit, X_test)
```

    ## $estimate
    ## [1] -1.0235563  2.3194540 -0.5776915 -0.6762113 -1.2336189
    ## 
    ## $ci
    ##            [,1]      [,2]
    ## [1,] -3.1989515 1.1518389
    ## [2,]  0.1016273 4.5372806
    ## [3,] -2.7163884 1.5610055
    ## [4,] -2.8682399 1.5158172
    ## [5,] -3.4223356 0.9550978
