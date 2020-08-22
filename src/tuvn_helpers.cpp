#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;

// ***********************************************************************
// Sampling methods for truncated normal samplers
// ***********************************************************************
// TUVN normal rejection sampling
double norm_rej(double a, double b)
{
  // first sample from a random normal
  double x = Rcpp::rnorm(1)[0];

  // keep going if p not in (a, b)
  while( (x < a) || (x > b) )
  {
    x = rnorm(1)[0];
  }
  return(x);
}

// TUVN Half-normal rejection sampling
double halfnorm_rej(double a, double b)
{
  double x = rnorm(1)[0];
  double abs_x = std::abs(x);

  // keep going if abs_p not in (a, b)
  while( (abs_x < a) || (abs_x > b) )
  {
    x = rnorm(1)[0];
    abs_x = std::abs(x);
  }
  return(abs_x);
}

// TUVN uniform rejection sampling
double unif_rej(double a, double b)
{
  while(true)
  {
    double x = Rcpp::runif(1, a, b)[0];
    double u = Rcpp::runif(1)[0];
    double rho;

    // cases for the ratio
    if( (0 >= a) & (0 <= b) ) {rho = exp(-1 * (x*x) / 2.0);}
    if (a > 0) {rho = exp( -1 * (x*x - a*a) / 2.0);}
    if (b < 0) {rho = exp(-1 * (x*x - b*b) / 2.0);}

    // accept step
    if(u <= rho) {return(x);}
  }
}

// TUVN exponential rejection sampling
double exp_rej(double a, double b)
{
  // using the optimal lambda define in the paper
  double lambda = (a + sqrt(a*a + 4.0)) / 2.0;

  // loop to generate the sample
  while(true)
  {
    double x = Rcpp::rweibull(1, 1, 1/lambda)[0] + a;
    double u = Rcpp::runif(1)[0];
    double rho = exp(-1 * (x - lambda) * (x - lambda) / 2.0);

    if(u <= rho & x < b) {return(x);}
  }
}

// *****************************************************************************
// lower bound for b 
// *****************************************************************************
// normal vs. uniform
// Calculates the lower bound in Lemma 2.1.1.4 in Li & Ghosh (2015). It is the
//     bound for b case 2.
double lower_b(double a)
{
  return(sqrt(2.0 * M_PI) + a);
}

// half-normal vs uniform
// Calculates the lower bound in 2.1.1.5 of Li and Ghosh. Called b_1(a) in the
//     case breakdown
double lower_b1(double a)
{
  return(sqrt(M_PI / 2.0) * exp(a*a / 2.0) + a);
}

// exponential vs uniform sampling
// Calculates the bound for b in lemma 2.1.1.6 of Li & Ghosh. Called b_2(a)
//     in the case breakdown.
double lower_b2(double a)
{
  double lambda = a / 2.0 + sqrt(a*a + 4.0) / 2.0;
  return(a + exp(0.5) / lambda * exp((a*a - a * sqrt(a*a + 4.0)) / 4.0));
}

// #**********************************************************************#
// Rejection sampling by case - list of cases given on page 4 of
// Li & Ghosh (2015)
// #**********************************************************************#
double sample_case1(double a, double b)
{
  double samp;
  if(a <= 0)
    samp = norm_rej(a, b);
  else if (a < 0.25696)
    samp = halfnorm_rej(a, b);
  else
    samp = exp_rej(a, b);
  return(samp);
}

double sample_case2(double a, double b)
{
  double samp;
  double this_lower_b = lower_b(a);

  if(b > this_lower_b)
    samp = norm_rej(a, b);
  else
    samp = unif_rej(a, b);

  return(samp);
}

double sample_case3(double a, double b)
{
  double samp;
  if(a < 0.25696)
  {
    double blower1 = lower_b1(a);
    if(b <= blower1)
      samp = unif_rej(a, b);
    else
      samp = halfnorm_rej(a, b);
  }
  else
  {
    double blower2 = lower_b2(a);
    if(b <= blower2)
      samp = unif_rej(a, b);
    else
      samp = exp_rej(a, b);
  }
  return(samp);
}

double sample_case4(double a, double b)
{
  double temp = sample_case1(-b, -a);
  return(-temp);
}

double sample_case5(double a, double b)
{
  double temp = sample_case3(-b, -a);
  return(-temp);
}

// Full Rejection sampling steps
double sample_tuvsn(double a, double b)
{
  double samp;
  if( (a == R_NegInf) || (b == R_PosInf) )
  {
    if(b == R_PosInf)
      samp = sample_case1(a, b);
    else
      samp = sample_case4(a, b);
  }
  else
  {
    if(a >= 0)
      samp = sample_case3(a, b);
    else if (b <= 0)
      samp = sample_case5(a, b);
    else
      samp = sample_case2(a, b);
  }
  return(samp);
}




