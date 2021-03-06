% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{probit_lasso_cavi}
\alias{probit_lasso_cavi}
\title{Univariate probit linear regression with a LASSO (double-exponential) prior
using the CAVI algorithm.}
\usage{
probit_lasso_cavi(
  y,
  X,
  n_iter = 1000L,
  verbose = TRUE,
  a_lambda2 = 0.1,
  b_lambda2 = 0.1,
  tol = 1e-04,
  type = 0L
)
}
\arguments{
\item{y}{Vector or responses (N by 1)}

\item{X}{Matrix of predictors (N by P)}

\item{n_iter}{Max number of iterations to run the algorithm for (default =
1000). A convergence warning is issues if the algorithm runs for the max
number of iterations.}

\item{verbose}{True or False. Do you want to print messages along the way?}

\item{a_lambda2}{Prior shape parameter for the coefficient precision
(shrinkage) term.}

\item{b_lambda2}{Prior rate parameter for the coefficient precision
(shrinkage) term.}

\item{type}{Correlation structure of the regression coefficients. Use 0 for
full correlation and 1 for independece assumption.}

\item{rel_tol}{Relative tolerance used for convergence. Convergence is
assesed using the evidence lower bound (ELBO) changes relative to five
iterations prior.}
}
\description{
Univariate probit linear regression with a LASSO (double-exponential) prior
using the CAVI algorithm.
}
