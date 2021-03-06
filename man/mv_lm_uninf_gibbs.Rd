% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/mv_lm_uninf_gibbs.R
\name{mv_lm_uninf_gibbs}
\alias{mv_lm_uninf_gibbs}
\title{Multivariate linear regression with a factor model}
\usage{
mv_lm_uninf_gibbs(Y, X, K = 2, n_iter = 10000, burn_in = 5000, verbose = TRUE)
}
\arguments{
\item{Y}{matrix of responses}

\item{X}{matrix of predictors to control for}

\item{K}{number of factors in the factor model}

\item{n_iter}{number of iterations to run the Gibbs sampler}

\item{burn_in}{number of iterations to drop for burn in.}

\item{verbose}{True or False. Print status of the sampler.}
}
\value{
A list with samples for the intercept term (\code{b0_mat}),
  regression coefficients (\code{B_array}), and upper triangular elements of
  the correlation matrix (\code{cor_mat_upper_tri})
}
\description{
Run a Gibbs sampler for the multivariate probit model.
}
