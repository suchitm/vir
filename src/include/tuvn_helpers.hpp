#ifndef __TRUNCNORM__
#define __TRUNCNORM__

double sample_tuvsn(double a, double b);
Eigen::VectorXd rtuvn(int n, double mean, double sd, double lower, double upper);

#endif // __TRUNCNORM__
