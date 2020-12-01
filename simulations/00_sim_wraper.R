rm(list = ls())

library(tidyverse)
library(glmnet)
library(vir)
library(statmod)
library(coda)
library(Hmisc)
library(PRROC)
library(fossil)
library(rstanarm)
library(stringr)

set.seed(42)

#--------------------------------------------------
# parameters used for both lm and probit funcitons
#--------------------------------------------------
gibbs_iter = 5000
seq_to_keep = 1000:gibbs_iter
svi_n_iter = 15000
cavi_n_iter = 1000
stan_iter = 100000
stan_rel_tol = 0.001
n_sim = 50

#**********************************************************************#
# CLONE THE GIT REPOSITORY TO YOUR COMPUTER AND WRITE PATH BELOW
#**********************************************************************#
REPO_PATH = "~/vir"
results_path = paste0(REPO_PATH, "/simulations/results/")

source(paste0(REPO_PATH, "/simulations/sim_helpers.R"))
source(paste0(REPO_PATH, "/simulations/sim_helpers_lm.R"))
source(paste0(REPO_PATH, "/simulations/sim_helpers_probit.R"))

# generate the data for the simulations
source(paste0(REPO_PATH, "/simulations/01_gen_sim_data.R"))

stop()

# linear model simualtions (Section 6.2)
# stan implementations do not run without error on Catalina 10.15.4
source(paste0(REPO_PATH, "/simulations/02_lm_sims_gibbs.R"))
source(paste0(REPO_PATH, "/simulations/02_lm_sims_glmnet.R"))
source(paste0(REPO_PATH, "/simulations/02_lm_sims_vir.R"))
# source(paste0(REPO_PATH, "/simulations/02_lm_sims_rstan.R"))

stop()

# probit simulations
source(paste0(REPO_PATH, "/simulations/03_probit_sims_gibbs.R"))
source(paste0(REPO_PATH, "/simulations/03_probit_sims_glmnet.R"))
source(paste0(REPO_PATH, "/simulations/03_probit_sims_vir.R"))
# source(paste0(REPO_PATH, "/simulations/03_probit_sims_rstan.R"))

stop()

# timing experiment simulations (Section 6.3)
source(paste0(REPO_PATH, "/simulations/04_run_timing_experiments.R"))


# process results
source(paste0(REPO_PATH, "/simulations/03_process_lm_sims.R"))
source(paste0(REPO_PATH, "/simulations/04_process_probit_sims.R"))
source(paste0(REPO_PATH, "/simulations/06_process_timing_results.R"))

# implementation section
source(paste0(REPO_PATH, "/simulations/07_implementation.R"))
