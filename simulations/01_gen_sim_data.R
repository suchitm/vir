set.seed(42)
#**********************************************************************#
# LM simulations
#**********************************************************************#
N = c(100, 1000, 5000)
P = 75
rho = 0.5
snr = 1
P = 75
rho = 0.5
snr = 1

data_path = paste0(REPO_PATH, "/data/")

rm_dats = paste0("rm -rf ", data_path, "/*")
system(rm_dats)

for(i in 1:n_sim)
{
  for(n in 1:length(N))
  {
    this_N = N[n]
    # simulate data
    this_dat = sim_data_lm(this_N, P, rho, snr)
    y_train = this_dat$y_train
    X_train = this_dat$X_train
    y_test = this_dat$y_test
    X_test = this_dat$X_test
    true_b = this_dat$b
    fname = paste0(data_path, "sim_lm_n", this_N, "_i", i, ".RData")
    save(this_dat, this_N, P, rho, snr, i, file = fname)
  }
}

#**********************************************************************#
# Probit simulations
#**********************************************************************#
N = c(500)
P = 50
rho = c(0.5)
sim_type = c("probit", "logit")

for(i in 1:n_sim)
{
  for(n in 1:length(N))
  {
    for(r in 1:length(sim_type))
    {
      this_N = N[n]
      this_type = sim_type[r]
      # simulate data
      this_dat = sim_data_binom(this_N, P, rho, type = this_type)
      fname = paste0(
        REPO_PATH, "/data/sim_probit_n", this_N, "_type", this_type, "_i", i,
        ".RData"
      )
      save(this_dat, this_N, P, this_type, i, file = fname)
    }
  }
}
