#ifndef __MULTIVARIATE__
#define __MULTIVARIATE__

#include "../typedefs.h"

// *****************************************************************************
// gibbs samplers
// *****************************************************************************
Eigen::MatrixXd mv_probit_gibbs_Z(
  Eigen::MatrixXi& Y, Eigen::MatrixXd& Eta, double& tau, int& N, int& M,
  Eigen::MatrixXd& Z
);

Eigen::MatrixXd mvlm_uninf_gibbs_mpsi(
  Eigen::MatrixXd& E_hat, Eigen::MatrixXd& mtheta, double& tau, int& N, int& K,
  Eigen::MatrixXd& mpsi
);

Eigen::VectorXd mvlm_uninf_gibbs_b0(
  Eigen::MatrixXd& E_hat, double& tau, int& N, int& M, Eigen::VectorXd& b0
);

Eigen::MatrixXd mvlm_uninf_gibbs_B(
  Eigen::MatrixXd& E_hat, Eigen::MatrixXd& X, double& tau, int& M,
  int& P, Eigen::MatrixXd& b
);

Eigen::MatrixXd mvlm_uninf_gibbs_mtheta(
  Eigen::MatrixXd& E_hat, Eigen::MatrixXd& mpsi, double& tau, int& M, int& K,
  Eigen::MatrixXd& mtheta
);

double mvlm_uninf_gibbs_tau(
  Eigen::MatrixXd& E_hat, int& N, int& M, double& a_tau, double& b_tau,
  double& tau
);

// *****************************************************************************
// others
// *****************************************************************************
Eigen::VectorXd cum_prod(Eigen::VectorXd& x);

#endif // __MULTIVARIATE__
