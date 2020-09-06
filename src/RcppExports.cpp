// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

// lm_hs_vi_elbo
double lm_hs_vi_elbo(Eigen::MatrixXd& X_s, Eigen::VectorXd& y_s, Rcpp::List& param_b0, Rcpp::List& param_b, Rcpp::List& param_tau, Rcpp::List& param_lambda, Rcpp::List& param_xi, Rcpp::List& param_gamma, Rcpp::List& param_nu, double& a_tau, double& b_tau, int& N, int& S, int& P);
RcppExport SEXP _vir_lm_hs_vi_elbo(SEXP X_sSEXP, SEXP y_sSEXP, SEXP param_b0SEXP, SEXP param_bSEXP, SEXP param_tauSEXP, SEXP param_lambdaSEXP, SEXP param_xiSEXP, SEXP param_gammaSEXP, SEXP param_nuSEXP, SEXP a_tauSEXP, SEXP b_tauSEXP, SEXP NSEXP, SEXP SSEXP, SEXP PSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type X_s(X_sSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type y_s(y_sSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_b0(param_b0SEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_b(param_bSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_tau(param_tauSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_lambda(param_lambdaSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_xi(param_xiSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_gamma(param_gammaSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_nu(param_nuSEXP);
    Rcpp::traits::input_parameter< double& >::type a_tau(a_tauSEXP);
    Rcpp::traits::input_parameter< double& >::type b_tau(b_tauSEXP);
    Rcpp::traits::input_parameter< int& >::type N(NSEXP);
    Rcpp::traits::input_parameter< int& >::type S(SSEXP);
    Rcpp::traits::input_parameter< int& >::type P(PSEXP);
    rcpp_result_gen = Rcpp::wrap(lm_hs_vi_elbo(X_s, y_s, param_b0, param_b, param_tau, param_lambda, param_xi, param_gamma, param_nu, a_tau, b_tau, N, S, P));
    return rcpp_result_gen;
END_RCPP
}
// lm_hs_cavi
Rcpp::List lm_hs_cavi(Eigen::VectorXd y, Eigen::MatrixXd X, int n_iter, bool verbose, double a_tau, double b_tau, double rel_tol, int type);
RcppExport SEXP _vir_lm_hs_cavi(SEXP ySEXP, SEXP XSEXP, SEXP n_iterSEXP, SEXP verboseSEXP, SEXP a_tauSEXP, SEXP b_tauSEXP, SEXP rel_tolSEXP, SEXP typeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type y(ySEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type n_iter(n_iterSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< double >::type a_tau(a_tauSEXP);
    Rcpp::traits::input_parameter< double >::type b_tau(b_tauSEXP);
    Rcpp::traits::input_parameter< double >::type rel_tol(rel_tolSEXP);
    Rcpp::traits::input_parameter< int >::type type(typeSEXP);
    rcpp_result_gen = Rcpp::wrap(lm_hs_cavi(y, X, n_iter, verbose, a_tau, b_tau, rel_tol, type));
    return rcpp_result_gen;
END_RCPP
}
// lm_hs_svi
Rcpp::List lm_hs_svi(Eigen::VectorXd y, Eigen::MatrixXd X, bool verbose, int n_iter, double a_tau, double b_tau, int type, int batch_size, double const_rhot, double omega, double kappa);
RcppExport SEXP _vir_lm_hs_svi(SEXP ySEXP, SEXP XSEXP, SEXP verboseSEXP, SEXP n_iterSEXP, SEXP a_tauSEXP, SEXP b_tauSEXP, SEXP typeSEXP, SEXP batch_sizeSEXP, SEXP const_rhotSEXP, SEXP omegaSEXP, SEXP kappaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type y(ySEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type X(XSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< int >::type n_iter(n_iterSEXP);
    Rcpp::traits::input_parameter< double >::type a_tau(a_tauSEXP);
    Rcpp::traits::input_parameter< double >::type b_tau(b_tauSEXP);
    Rcpp::traits::input_parameter< int >::type type(typeSEXP);
    Rcpp::traits::input_parameter< int >::type batch_size(batch_sizeSEXP);
    Rcpp::traits::input_parameter< double >::type const_rhot(const_rhotSEXP);
    Rcpp::traits::input_parameter< double >::type omega(omegaSEXP);
    Rcpp::traits::input_parameter< double >::type kappa(kappaSEXP);
    rcpp_result_gen = Rcpp::wrap(lm_hs_svi(y, X, verbose, n_iter, a_tau, b_tau, type, batch_size, const_rhot, omega, kappa));
    return rcpp_result_gen;
END_RCPP
}
// lm_lasso_cavi
Rcpp::List lm_lasso_cavi(Eigen::VectorXd y, Eigen::MatrixXd X, bool verbose, int n_iter, double a_tau, double b_tau, double a_lambda2, double b_lambda2, double rel_tol, int type);
RcppExport SEXP _vir_lm_lasso_cavi(SEXP ySEXP, SEXP XSEXP, SEXP verboseSEXP, SEXP n_iterSEXP, SEXP a_tauSEXP, SEXP b_tauSEXP, SEXP a_lambda2SEXP, SEXP b_lambda2SEXP, SEXP rel_tolSEXP, SEXP typeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type y(ySEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type X(XSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< int >::type n_iter(n_iterSEXP);
    Rcpp::traits::input_parameter< double >::type a_tau(a_tauSEXP);
    Rcpp::traits::input_parameter< double >::type b_tau(b_tauSEXP);
    Rcpp::traits::input_parameter< double >::type a_lambda2(a_lambda2SEXP);
    Rcpp::traits::input_parameter< double >::type b_lambda2(b_lambda2SEXP);
    Rcpp::traits::input_parameter< double >::type rel_tol(rel_tolSEXP);
    Rcpp::traits::input_parameter< int >::type type(typeSEXP);
    rcpp_result_gen = Rcpp::wrap(lm_lasso_cavi(y, X, verbose, n_iter, a_tau, b_tau, a_lambda2, b_lambda2, rel_tol, type));
    return rcpp_result_gen;
END_RCPP
}
// lm_lasso_svi
Rcpp::List lm_lasso_svi(Eigen::VectorXd y, Eigen::MatrixXd X, bool verbose, int n_iter, double a_tau, double b_tau, double a_lambda2, double b_lambda2, int type, int batch_size, double const_rhot, double omega, double kappa);
RcppExport SEXP _vir_lm_lasso_svi(SEXP ySEXP, SEXP XSEXP, SEXP verboseSEXP, SEXP n_iterSEXP, SEXP a_tauSEXP, SEXP b_tauSEXP, SEXP a_lambda2SEXP, SEXP b_lambda2SEXP, SEXP typeSEXP, SEXP batch_sizeSEXP, SEXP const_rhotSEXP, SEXP omegaSEXP, SEXP kappaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type y(ySEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type X(XSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< int >::type n_iter(n_iterSEXP);
    Rcpp::traits::input_parameter< double >::type a_tau(a_tauSEXP);
    Rcpp::traits::input_parameter< double >::type b_tau(b_tauSEXP);
    Rcpp::traits::input_parameter< double >::type a_lambda2(a_lambda2SEXP);
    Rcpp::traits::input_parameter< double >::type b_lambda2(b_lambda2SEXP);
    Rcpp::traits::input_parameter< int >::type type(typeSEXP);
    Rcpp::traits::input_parameter< int >::type batch_size(batch_sizeSEXP);
    Rcpp::traits::input_parameter< double >::type const_rhot(const_rhotSEXP);
    Rcpp::traits::input_parameter< double >::type omega(omegaSEXP);
    Rcpp::traits::input_parameter< double >::type kappa(kappaSEXP);
    rcpp_result_gen = Rcpp::wrap(lm_lasso_svi(y, X, verbose, n_iter, a_tau, b_tau, a_lambda2, b_lambda2, type, batch_size, const_rhot, omega, kappa));
    return rcpp_result_gen;
END_RCPP
}
// lm_ridge_cavi
Rcpp::List lm_ridge_cavi(Eigen::VectorXd y, Eigen::MatrixXd X, int n_iter, bool verbose, double a_tau, double b_tau, double a_lambda, double b_lambda, double rel_tol, int type);
RcppExport SEXP _vir_lm_ridge_cavi(SEXP ySEXP, SEXP XSEXP, SEXP n_iterSEXP, SEXP verboseSEXP, SEXP a_tauSEXP, SEXP b_tauSEXP, SEXP a_lambdaSEXP, SEXP b_lambdaSEXP, SEXP rel_tolSEXP, SEXP typeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type y(ySEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type n_iter(n_iterSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< double >::type a_tau(a_tauSEXP);
    Rcpp::traits::input_parameter< double >::type b_tau(b_tauSEXP);
    Rcpp::traits::input_parameter< double >::type a_lambda(a_lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type b_lambda(b_lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type rel_tol(rel_tolSEXP);
    Rcpp::traits::input_parameter< int >::type type(typeSEXP);
    rcpp_result_gen = Rcpp::wrap(lm_ridge_cavi(y, X, n_iter, verbose, a_tau, b_tau, a_lambda, b_lambda, rel_tol, type));
    return rcpp_result_gen;
END_RCPP
}
// lm_ridge_svi
Rcpp::List lm_ridge_svi(Eigen::VectorXd y, Eigen::MatrixXd X, int n_iter, bool verbose, double a_tau, double b_tau, double a_lambda, double b_lambda, int type, int batch_size, double const_rhot, double omega, double kappa);
RcppExport SEXP _vir_lm_ridge_svi(SEXP ySEXP, SEXP XSEXP, SEXP n_iterSEXP, SEXP verboseSEXP, SEXP a_tauSEXP, SEXP b_tauSEXP, SEXP a_lambdaSEXP, SEXP b_lambdaSEXP, SEXP typeSEXP, SEXP batch_sizeSEXP, SEXP const_rhotSEXP, SEXP omegaSEXP, SEXP kappaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd >::type y(ySEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type n_iter(n_iterSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< double >::type a_tau(a_tauSEXP);
    Rcpp::traits::input_parameter< double >::type b_tau(b_tauSEXP);
    Rcpp::traits::input_parameter< double >::type a_lambda(a_lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type b_lambda(b_lambdaSEXP);
    Rcpp::traits::input_parameter< int >::type type(typeSEXP);
    Rcpp::traits::input_parameter< int >::type batch_size(batch_sizeSEXP);
    Rcpp::traits::input_parameter< double >::type const_rhot(const_rhotSEXP);
    Rcpp::traits::input_parameter< double >::type omega(omegaSEXP);
    Rcpp::traits::input_parameter< double >::type kappa(kappaSEXP);
    rcpp_result_gen = Rcpp::wrap(lm_ridge_svi(y, X, n_iter, verbose, a_tau, b_tau, a_lambda, b_lambda, type, batch_size, const_rhot, omega, kappa));
    return rcpp_result_gen;
END_RCPP
}
// mv_probit_gibbs_Z
Eigen::MatrixXd mv_probit_gibbs_Z(Eigen::MatrixXi& Y, Eigen::MatrixXd& Eta, Eigen::VectorXd& tau, int& N, int& M, Eigen::MatrixXd& Z);
RcppExport SEXP _vir_mv_probit_gibbs_Z(SEXP YSEXP, SEXP EtaSEXP, SEXP tauSEXP, SEXP NSEXP, SEXP MSEXP, SEXP ZSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXi& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type Eta(EtaSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< int& >::type N(NSEXP);
    Rcpp::traits::input_parameter< int& >::type M(MSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type Z(ZSEXP);
    rcpp_result_gen = Rcpp::wrap(mv_probit_gibbs_Z(Y, Eta, tau, N, M, Z));
    return rcpp_result_gen;
END_RCPP
}
// mvlm_uninf_gibbs_b0
Eigen::VectorXd mvlm_uninf_gibbs_b0(Eigen::MatrixXd& E_hat, Eigen::VectorXd& tau, int& N, int& M, Eigen::VectorXd& b0);
RcppExport SEXP _vir_mvlm_uninf_gibbs_b0(SEXP E_hatSEXP, SEXP tauSEXP, SEXP NSEXP, SEXP MSEXP, SEXP b0SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type E_hat(E_hatSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< int& >::type N(NSEXP);
    Rcpp::traits::input_parameter< int& >::type M(MSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type b0(b0SEXP);
    rcpp_result_gen = Rcpp::wrap(mvlm_uninf_gibbs_b0(E_hat, tau, N, M, b0));
    return rcpp_result_gen;
END_RCPP
}
// mvlm_uninf_gibbs_B
Eigen::MatrixXd mvlm_uninf_gibbs_B(Eigen::MatrixXd& E_hat, Eigen::MatrixXd& X, Eigen::VectorXd& tau, int& M, int& P, Eigen::MatrixXd& b);
RcppExport SEXP _vir_mvlm_uninf_gibbs_B(SEXP E_hatSEXP, SEXP XSEXP, SEXP tauSEXP, SEXP MSEXP, SEXP PSEXP, SEXP bSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type E_hat(E_hatSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< int& >::type M(MSEXP);
    Rcpp::traits::input_parameter< int& >::type P(PSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type b(bSEXP);
    rcpp_result_gen = Rcpp::wrap(mvlm_uninf_gibbs_B(E_hat, X, tau, M, P, b));
    return rcpp_result_gen;
END_RCPP
}
// mvlm_uninf_gibbs_mtheta
Eigen::MatrixXd mvlm_uninf_gibbs_mtheta(Eigen::MatrixXd& E_hat, Eigen::MatrixXd& mphi, Eigen::MatrixXd& mgamma, Eigen::VectorXd& tau, Eigen::VectorXd& lambda, int& M, int& K, Eigen::MatrixXd& mtheta);
RcppExport SEXP _vir_mvlm_uninf_gibbs_mtheta(SEXP E_hatSEXP, SEXP mphiSEXP, SEXP mgammaSEXP, SEXP tauSEXP, SEXP lambdaSEXP, SEXP MSEXP, SEXP KSEXP, SEXP mthetaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type E_hat(E_hatSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type mphi(mphiSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type mgamma(mgammaSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< int& >::type M(MSEXP);
    Rcpp::traits::input_parameter< int& >::type K(KSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type mtheta(mthetaSEXP);
    rcpp_result_gen = Rcpp::wrap(mvlm_uninf_gibbs_mtheta(E_hat, mphi, mgamma, tau, lambda, M, K, mtheta));
    return rcpp_result_gen;
END_RCPP
}
// mvlm_uninf_gibbs_mphi
Eigen::MatrixXd mvlm_uninf_gibbs_mphi(Eigen::MatrixXd& E_hat, Eigen::MatrixXd& mtheta, Eigen::VectorXd& tau, int& N, int& K, Eigen::MatrixXd& mphi);
RcppExport SEXP _vir_mvlm_uninf_gibbs_mphi(SEXP E_hatSEXP, SEXP mthetaSEXP, SEXP tauSEXP, SEXP NSEXP, SEXP KSEXP, SEXP mphiSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type E_hat(E_hatSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type mtheta(mthetaSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< int& >::type N(NSEXP);
    Rcpp::traits::input_parameter< int& >::type K(KSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type mphi(mphiSEXP);
    rcpp_result_gen = Rcpp::wrap(mvlm_uninf_gibbs_mphi(E_hat, mtheta, tau, N, K, mphi));
    return rcpp_result_gen;
END_RCPP
}
// mvlm_uninf_gibbs_tau
Eigen::VectorXd mvlm_uninf_gibbs_tau(Eigen::MatrixXd& E_hat, int& N, int& M, double& a_tau, double& b_tau, Eigen::VectorXd& tau);
RcppExport SEXP _vir_mvlm_uninf_gibbs_tau(SEXP E_hatSEXP, SEXP NSEXP, SEXP MSEXP, SEXP a_tauSEXP, SEXP b_tauSEXP, SEXP tauSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type E_hat(E_hatSEXP);
    Rcpp::traits::input_parameter< int& >::type N(NSEXP);
    Rcpp::traits::input_parameter< int& >::type M(MSEXP);
    Rcpp::traits::input_parameter< double& >::type a_tau(a_tauSEXP);
    Rcpp::traits::input_parameter< double& >::type b_tau(b_tauSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type tau(tauSEXP);
    rcpp_result_gen = Rcpp::wrap(mvlm_uninf_gibbs_tau(E_hat, N, M, a_tau, b_tau, tau));
    return rcpp_result_gen;
END_RCPP
}
// mvlm_uninf_gibbs_mgamma
Eigen::MatrixXd mvlm_uninf_gibbs_mgamma(Eigen::MatrixXd& mtheta, Eigen::VectorXd& lambda, int& M, int& K, double& nu, Eigen::MatrixXd& mgamma);
RcppExport SEXP _vir_mvlm_uninf_gibbs_mgamma(SEXP mthetaSEXP, SEXP lambdaSEXP, SEXP MSEXP, SEXP KSEXP, SEXP nuSEXP, SEXP mgammaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type mtheta(mthetaSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< int& >::type M(MSEXP);
    Rcpp::traits::input_parameter< int& >::type K(KSEXP);
    Rcpp::traits::input_parameter< double& >::type nu(nuSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type mgamma(mgammaSEXP);
    rcpp_result_gen = Rcpp::wrap(mvlm_uninf_gibbs_mgamma(mtheta, lambda, M, K, nu, mgamma));
    return rcpp_result_gen;
END_RCPP
}
// mvlm_uninf_gibbs_xi
Eigen::VectorXd mvlm_uninf_gibbs_xi(Eigen::MatrixXd& mtheta, Eigen::MatrixXd& mgamma, Eigen::VectorXd& lambda, int& M, int& K, double& a1, double& a2, Eigen::VectorXd& xi);
RcppExport SEXP _vir_mvlm_uninf_gibbs_xi(SEXP mthetaSEXP, SEXP mgammaSEXP, SEXP lambdaSEXP, SEXP MSEXP, SEXP KSEXP, SEXP a1SEXP, SEXP a2SEXP, SEXP xiSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type mtheta(mthetaSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type mgamma(mgammaSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< int& >::type M(MSEXP);
    Rcpp::traits::input_parameter< int& >::type K(KSEXP);
    Rcpp::traits::input_parameter< double& >::type a1(a1SEXP);
    Rcpp::traits::input_parameter< double& >::type a2(a2SEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type xi(xiSEXP);
    rcpp_result_gen = Rcpp::wrap(mvlm_uninf_gibbs_xi(mtheta, mgamma, lambda, M, K, a1, a2, xi));
    return rcpp_result_gen;
END_RCPP
}
// mv_probit_vi_z
Rcpp::List mv_probit_vi_z(Eigen::MatrixXd& Eta, Rcpp::List& param_tau, int& S, int& M, Rcpp::List& param_z);
RcppExport SEXP _vir_mv_probit_vi_z(SEXP EtaSEXP, SEXP param_tauSEXP, SEXP SSEXP, SEXP MSEXP, SEXP param_zSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type Eta(EtaSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_tau(param_tauSEXP);
    Rcpp::traits::input_parameter< int& >::type S(SSEXP);
    Rcpp::traits::input_parameter< int& >::type M(MSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_z(param_zSEXP);
    rcpp_result_gen = Rcpp::wrap(mv_probit_vi_z(Eta, param_tau, S, M, param_z));
    return rcpp_result_gen;
END_RCPP
}
// mvlm_vi_phi
Rcpp::List mvlm_vi_phi(Eigen::MatrixXd& E_hat, Rcpp::List& param_theta, Rcpp::List& param_tau, int& S, int& M, int& K, Rcpp::List& param_phi);
RcppExport SEXP _vir_mvlm_vi_phi(SEXP E_hatSEXP, SEXP param_thetaSEXP, SEXP param_tauSEXP, SEXP SSEXP, SEXP MSEXP, SEXP KSEXP, SEXP param_phiSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type E_hat(E_hatSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_theta(param_thetaSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_tau(param_tauSEXP);
    Rcpp::traits::input_parameter< int& >::type S(SSEXP);
    Rcpp::traits::input_parameter< int& >::type M(MSEXP);
    Rcpp::traits::input_parameter< int& >::type K(KSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_phi(param_phiSEXP);
    rcpp_result_gen = Rcpp::wrap(mvlm_vi_phi(E_hat, param_theta, param_tau, S, M, K, param_phi));
    return rcpp_result_gen;
END_RCPP
}
// mvlm_vi_theta
Rcpp::List mvlm_vi_theta(Eigen::MatrixXd& E_hat, Rcpp::List& param_phi, Rcpp::List& param_tau, Rcpp::List& param_gamma, Eigen::VectorXd& mu_lambda, int& N, int& M, int& S, int& K, Rcpp::List& param_theta);
RcppExport SEXP _vir_mvlm_vi_theta(SEXP E_hatSEXP, SEXP param_phiSEXP, SEXP param_tauSEXP, SEXP param_gammaSEXP, SEXP mu_lambdaSEXP, SEXP NSEXP, SEXP MSEXP, SEXP SSEXP, SEXP KSEXP, SEXP param_thetaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type E_hat(E_hatSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_phi(param_phiSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_tau(param_tauSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_gamma(param_gammaSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type mu_lambda(mu_lambdaSEXP);
    Rcpp::traits::input_parameter< int& >::type N(NSEXP);
    Rcpp::traits::input_parameter< int& >::type M(MSEXP);
    Rcpp::traits::input_parameter< int& >::type S(SSEXP);
    Rcpp::traits::input_parameter< int& >::type K(KSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_theta(param_thetaSEXP);
    rcpp_result_gen = Rcpp::wrap(mvlm_vi_theta(E_hat, param_phi, param_tau, param_gamma, mu_lambda, N, M, S, K, param_theta));
    return rcpp_result_gen;
END_RCPP
}
// mvlm_vi_b0
Rcpp::List mvlm_vi_b0(Eigen::MatrixXd& E_hat, Rcpp::List& param_tau, int& N, int& M, int& S, Rcpp::List& param_b0);
RcppExport SEXP _vir_mvlm_vi_b0(SEXP E_hatSEXP, SEXP param_tauSEXP, SEXP NSEXP, SEXP MSEXP, SEXP SSEXP, SEXP param_b0SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type E_hat(E_hatSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_tau(param_tauSEXP);
    Rcpp::traits::input_parameter< int& >::type N(NSEXP);
    Rcpp::traits::input_parameter< int& >::type M(MSEXP);
    Rcpp::traits::input_parameter< int& >::type S(SSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_b0(param_b0SEXP);
    rcpp_result_gen = Rcpp::wrap(mvlm_vi_b0(E_hat, param_tau, N, M, S, param_b0));
    return rcpp_result_gen;
END_RCPP
}
// mvlm_vi_b
Rcpp::List mvlm_vi_b(Eigen::MatrixXd& E_hat, Eigen::MatrixXd& X, Rcpp::List& param_tau, int& N, int& M, int& P, int& S, Rcpp::List& param_b);
RcppExport SEXP _vir_mvlm_vi_b(SEXP E_hatSEXP, SEXP XSEXP, SEXP param_tauSEXP, SEXP NSEXP, SEXP MSEXP, SEXP PSEXP, SEXP SSEXP, SEXP param_bSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type E_hat(E_hatSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_tau(param_tauSEXP);
    Rcpp::traits::input_parameter< int& >::type N(NSEXP);
    Rcpp::traits::input_parameter< int& >::type M(MSEXP);
    Rcpp::traits::input_parameter< int& >::type P(PSEXP);
    Rcpp::traits::input_parameter< int& >::type S(SSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_b(param_bSEXP);
    rcpp_result_gen = Rcpp::wrap(mvlm_vi_b(E_hat, X, param_tau, N, M, P, S, param_b));
    return rcpp_result_gen;
END_RCPP
}
// mvlm_vi_tau
Rcpp::List mvlm_vi_tau(Eigen::MatrixXd& E_hat, Eigen::MatrixXd& X, Rcpp::List& param_b0, Rcpp::List& param_b, Rcpp::List& param_phi, Rcpp::List& param_theta, int& N, int& M, int& P, int& S, int& K, double& a_tau, double& b_tau, Rcpp::List& param_tau);
RcppExport SEXP _vir_mvlm_vi_tau(SEXP E_hatSEXP, SEXP XSEXP, SEXP param_b0SEXP, SEXP param_bSEXP, SEXP param_phiSEXP, SEXP param_thetaSEXP, SEXP NSEXP, SEXP MSEXP, SEXP PSEXP, SEXP SSEXP, SEXP KSEXP, SEXP a_tauSEXP, SEXP b_tauSEXP, SEXP param_tauSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type E_hat(E_hatSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_b0(param_b0SEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_b(param_bSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_phi(param_phiSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_theta(param_thetaSEXP);
    Rcpp::traits::input_parameter< int& >::type N(NSEXP);
    Rcpp::traits::input_parameter< int& >::type M(MSEXP);
    Rcpp::traits::input_parameter< int& >::type P(PSEXP);
    Rcpp::traits::input_parameter< int& >::type S(SSEXP);
    Rcpp::traits::input_parameter< int& >::type K(KSEXP);
    Rcpp::traits::input_parameter< double& >::type a_tau(a_tauSEXP);
    Rcpp::traits::input_parameter< double& >::type b_tau(b_tauSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_tau(param_tauSEXP);
    rcpp_result_gen = Rcpp::wrap(mvlm_vi_tau(E_hat, X, param_b0, param_b, param_phi, param_theta, N, M, P, S, K, a_tau, b_tau, param_tau));
    return rcpp_result_gen;
END_RCPP
}
// mvlm_vi_gamma
Rcpp::List mvlm_vi_gamma(Rcpp::List& param_theta, Eigen::VectorXd& mu_lambda, int& M, int& K, double& nu, Rcpp::List& param_gamma);
RcppExport SEXP _vir_mvlm_vi_gamma(SEXP param_thetaSEXP, SEXP mu_lambdaSEXP, SEXP MSEXP, SEXP KSEXP, SEXP nuSEXP, SEXP param_gammaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_theta(param_thetaSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type mu_lambda(mu_lambdaSEXP);
    Rcpp::traits::input_parameter< int& >::type M(MSEXP);
    Rcpp::traits::input_parameter< int& >::type K(KSEXP);
    Rcpp::traits::input_parameter< double& >::type nu(nuSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_gamma(param_gammaSEXP);
    rcpp_result_gen = Rcpp::wrap(mvlm_vi_gamma(param_theta, mu_lambda, M, K, nu, param_gamma));
    return rcpp_result_gen;
END_RCPP
}
// mvlm_vi_xi
Rcpp::List mvlm_vi_xi(Rcpp::List& param_gamma, Rcpp::List& param_theta, int& M, int& K, Eigen::VectorXd& mu_lambda, double& a1, double& a2, Rcpp::List& param_xi, bool svb);
RcppExport SEXP _vir_mvlm_vi_xi(SEXP param_gammaSEXP, SEXP param_thetaSEXP, SEXP MSEXP, SEXP KSEXP, SEXP mu_lambdaSEXP, SEXP a1SEXP, SEXP a2SEXP, SEXP param_xiSEXP, SEXP svbSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_gamma(param_gammaSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_theta(param_thetaSEXP);
    Rcpp::traits::input_parameter< int& >::type M(MSEXP);
    Rcpp::traits::input_parameter< int& >::type K(KSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type mu_lambda(mu_lambdaSEXP);
    Rcpp::traits::input_parameter< double& >::type a1(a1SEXP);
    Rcpp::traits::input_parameter< double& >::type a2(a2SEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_xi(param_xiSEXP);
    Rcpp::traits::input_parameter< bool >::type svb(svbSEXP);
    rcpp_result_gen = Rcpp::wrap(mvlm_vi_xi(param_gamma, param_theta, M, K, mu_lambda, a1, a2, param_xi, svb));
    return rcpp_result_gen;
END_RCPP
}
// mv_probit_vi_z_psi
Rcpp::List mv_probit_vi_z_psi(Eigen::MatrixXi& Y_s, Eigen::MatrixXd& E_hat, Rcpp::List& param_theta, int& S, int& M, int& K, Rcpp::List& param_psi, Rcpp::List& param_z);
RcppExport SEXP _vir_mv_probit_vi_z_psi(SEXP Y_sSEXP, SEXP E_hatSEXP, SEXP param_thetaSEXP, SEXP SSEXP, SEXP MSEXP, SEXP KSEXP, SEXP param_psiSEXP, SEXP param_zSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXi& >::type Y_s(Y_sSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type E_hat(E_hatSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_theta(param_thetaSEXP);
    Rcpp::traits::input_parameter< int& >::type S(SSEXP);
    Rcpp::traits::input_parameter< int& >::type M(MSEXP);
    Rcpp::traits::input_parameter< int& >::type K(KSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_psi(param_psiSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_z(param_zSEXP);
    rcpp_result_gen = Rcpp::wrap(mv_probit_vi_z_psi(Y_s, E_hat, param_theta, S, M, K, param_psi, param_z));
    return rcpp_result_gen;
END_RCPP
}
// probit_gibbs_z
Eigen::VectorXd probit_gibbs_z(Eigen::VectorXi& y, Eigen::VectorXd& eta, int& N, Eigen::VectorXd& z);
RcppExport SEXP _vir_probit_gibbs_z(SEXP ySEXP, SEXP etaSEXP, SEXP NSEXP, SEXP zSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXi& >::type y(ySEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< int& >::type N(NSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type z(zSEXP);
    rcpp_result_gen = Rcpp::wrap(probit_gibbs_z(y, eta, N, z));
    return rcpp_result_gen;
END_RCPP
}
// probit_gibbs_b0
double probit_gibbs_b0(Eigen::VectorXd& ehat, int& N, double& b0);
RcppExport SEXP _vir_probit_gibbs_b0(SEXP ehatSEXP, SEXP NSEXP, SEXP b0SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type ehat(ehatSEXP);
    Rcpp::traits::input_parameter< int& >::type N(NSEXP);
    Rcpp::traits::input_parameter< double& >::type b0(b0SEXP);
    rcpp_result_gen = Rcpp::wrap(probit_gibbs_b0(ehat, N, b0));
    return rcpp_result_gen;
END_RCPP
}
// probit_gibbs_b
Eigen::VectorXd probit_gibbs_b(Eigen::MatrixXd& X, Eigen::VectorXd& ehat, Eigen::MatrixXd& prior_mat, int& P, Eigen::VectorXd& b);
RcppExport SEXP _vir_probit_gibbs_b(SEXP XSEXP, SEXP ehatSEXP, SEXP prior_matSEXP, SEXP PSEXP, SEXP bSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type ehat(ehatSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type prior_mat(prior_matSEXP);
    Rcpp::traits::input_parameter< int& >::type P(PSEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type b(bSEXP);
    rcpp_result_gen = Rcpp::wrap(probit_gibbs_b(X, ehat, prior_mat, P, b));
    return rcpp_result_gen;
END_RCPP
}
// probit_log_lik
double probit_log_lik(Eigen::VectorXi& y, Eigen::MatrixXd& X, double& b0, Eigen::VectorXd& b, int& N);
RcppExport SEXP _vir_probit_log_lik(SEXP ySEXP, SEXP XSEXP, SEXP b0SEXP, SEXP bSEXP, SEXP NSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::VectorXi& >::type y(ySEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type X(XSEXP);
    Rcpp::traits::input_parameter< double& >::type b0(b0SEXP);
    Rcpp::traits::input_parameter< Eigen::VectorXd& >::type b(bSEXP);
    Rcpp::traits::input_parameter< int& >::type N(NSEXP);
    rcpp_result_gen = Rcpp::wrap(probit_log_lik(y, X, b0, b, N));
    return rcpp_result_gen;
END_RCPP
}
// probit_vi_z
Rcpp::List probit_vi_z(Eigen::MatrixXd& X_s, Rcpp::List& param_b0, Rcpp::List& param_b, int& S, Rcpp::List& param_z);
RcppExport SEXP _vir_probit_vi_z(SEXP X_sSEXP, SEXP param_b0SEXP, SEXP param_bSEXP, SEXP SSEXP, SEXP param_zSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type X_s(X_sSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_b0(param_b0SEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_b(param_bSEXP);
    Rcpp::traits::input_parameter< int& >::type S(SSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_z(param_zSEXP);
    rcpp_result_gen = Rcpp::wrap(probit_vi_z(X_s, param_b0, param_b, S, param_z));
    return rcpp_result_gen;
END_RCPP
}
// probit_vi_b0
Rcpp::List probit_vi_b0(Eigen::MatrixXd& X_s, Rcpp::List& param_z, Rcpp::List& param_b, int& N, int& S, Rcpp::List& param_b0);
RcppExport SEXP _vir_probit_vi_b0(SEXP X_sSEXP, SEXP param_zSEXP, SEXP param_bSEXP, SEXP NSEXP, SEXP SSEXP, SEXP param_b0SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type X_s(X_sSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_z(param_zSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_b(param_bSEXP);
    Rcpp::traits::input_parameter< int& >::type N(NSEXP);
    Rcpp::traits::input_parameter< int& >::type S(SSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_b0(param_b0SEXP);
    rcpp_result_gen = Rcpp::wrap(probit_vi_b0(X_s, param_z, param_b, N, S, param_b0));
    return rcpp_result_gen;
END_RCPP
}
// probit_vi_b
Rcpp::List probit_vi_b(Eigen::MatrixXd& X_s, Rcpp::List& param_z, Rcpp::List& param_b0, Eigen::MatrixXd& mu_prior_mat, int& N, int& S, int& P, int& type, bool cavi, Rcpp::List& param_b);
RcppExport SEXP _vir_probit_vi_b(SEXP X_sSEXP, SEXP param_zSEXP, SEXP param_b0SEXP, SEXP mu_prior_matSEXP, SEXP NSEXP, SEXP SSEXP, SEXP PSEXP, SEXP typeSEXP, SEXP caviSEXP, SEXP param_bSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type X_s(X_sSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_z(param_zSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_b0(param_b0SEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type mu_prior_mat(mu_prior_matSEXP);
    Rcpp::traits::input_parameter< int& >::type N(NSEXP);
    Rcpp::traits::input_parameter< int& >::type S(SSEXP);
    Rcpp::traits::input_parameter< int& >::type P(PSEXP);
    Rcpp::traits::input_parameter< int& >::type type(typeSEXP);
    Rcpp::traits::input_parameter< bool >::type cavi(caviSEXP);
    Rcpp::traits::input_parameter< Rcpp::List& >::type param_b(param_bSEXP);
    rcpp_result_gen = Rcpp::wrap(probit_vi_b(X_s, param_z, param_b0, mu_prior_mat, N, S, P, type, cavi, param_b));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_vir_lm_hs_vi_elbo", (DL_FUNC) &_vir_lm_hs_vi_elbo, 14},
    {"_vir_lm_hs_cavi", (DL_FUNC) &_vir_lm_hs_cavi, 8},
    {"_vir_lm_hs_svi", (DL_FUNC) &_vir_lm_hs_svi, 11},
    {"_vir_lm_lasso_cavi", (DL_FUNC) &_vir_lm_lasso_cavi, 10},
    {"_vir_lm_lasso_svi", (DL_FUNC) &_vir_lm_lasso_svi, 13},
    {"_vir_lm_ridge_cavi", (DL_FUNC) &_vir_lm_ridge_cavi, 10},
    {"_vir_lm_ridge_svi", (DL_FUNC) &_vir_lm_ridge_svi, 13},
    {"_vir_mv_probit_gibbs_Z", (DL_FUNC) &_vir_mv_probit_gibbs_Z, 6},
    {"_vir_mvlm_uninf_gibbs_b0", (DL_FUNC) &_vir_mvlm_uninf_gibbs_b0, 5},
    {"_vir_mvlm_uninf_gibbs_B", (DL_FUNC) &_vir_mvlm_uninf_gibbs_B, 6},
    {"_vir_mvlm_uninf_gibbs_mtheta", (DL_FUNC) &_vir_mvlm_uninf_gibbs_mtheta, 8},
    {"_vir_mvlm_uninf_gibbs_mphi", (DL_FUNC) &_vir_mvlm_uninf_gibbs_mphi, 6},
    {"_vir_mvlm_uninf_gibbs_tau", (DL_FUNC) &_vir_mvlm_uninf_gibbs_tau, 6},
    {"_vir_mvlm_uninf_gibbs_mgamma", (DL_FUNC) &_vir_mvlm_uninf_gibbs_mgamma, 6},
    {"_vir_mvlm_uninf_gibbs_xi", (DL_FUNC) &_vir_mvlm_uninf_gibbs_xi, 8},
    {"_vir_mv_probit_vi_z", (DL_FUNC) &_vir_mv_probit_vi_z, 5},
    {"_vir_mvlm_vi_phi", (DL_FUNC) &_vir_mvlm_vi_phi, 7},
    {"_vir_mvlm_vi_theta", (DL_FUNC) &_vir_mvlm_vi_theta, 10},
    {"_vir_mvlm_vi_b0", (DL_FUNC) &_vir_mvlm_vi_b0, 6},
    {"_vir_mvlm_vi_b", (DL_FUNC) &_vir_mvlm_vi_b, 8},
    {"_vir_mvlm_vi_tau", (DL_FUNC) &_vir_mvlm_vi_tau, 14},
    {"_vir_mvlm_vi_gamma", (DL_FUNC) &_vir_mvlm_vi_gamma, 6},
    {"_vir_mvlm_vi_xi", (DL_FUNC) &_vir_mvlm_vi_xi, 9},
    {"_vir_mv_probit_vi_z_psi", (DL_FUNC) &_vir_mv_probit_vi_z_psi, 8},
    {"_vir_probit_gibbs_z", (DL_FUNC) &_vir_probit_gibbs_z, 4},
    {"_vir_probit_gibbs_b0", (DL_FUNC) &_vir_probit_gibbs_b0, 3},
    {"_vir_probit_gibbs_b", (DL_FUNC) &_vir_probit_gibbs_b, 5},
    {"_vir_probit_log_lik", (DL_FUNC) &_vir_probit_log_lik, 5},
    {"_vir_probit_vi_z", (DL_FUNC) &_vir_probit_vi_z, 5},
    {"_vir_probit_vi_b0", (DL_FUNC) &_vir_probit_vi_b0, 6},
    {"_vir_probit_vi_b", (DL_FUNC) &_vir_probit_vi_b, 10},
    {NULL, NULL, 0}
};

RcppExport void R_init_vir(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
