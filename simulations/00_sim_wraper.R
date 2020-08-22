rm(list = ls())

library(tidyverse)
library(glmnet)
library(fastbayes)
library(statmod)
library(coda)
library(Hmisc)
library(PRROC)
library(fossil)

set.seed(42)

#**********************************************************************#
# CLONE THE GIT REPOSITORY TO YOUR COMPUTER AND WRITE PATH BELOW
#**********************************************************************#
REPO_PATH = "~/vir"
results_path = paste0(REPO_PATH, "/simulations/results/")

source(paste0(REPO_PATH, "/simulations/sim_helpers.R"))
source(paste0(REPO_PATH, "/simulations/sim_helpers_lm.R"))
source(paste0(REPO_PATH, "/simulations/sim_helpers_probit.R"))

# linear model simulations (Section 6.2)
source(paste0(REPO_PATH, "/simulations/01_run_lm_sims.R"))

# probit simulations (Section 6.3)
source(paste0(REPO_PATH, "/simulations/02_run_probit_sims.R"))

# timing experiment simulations (Section 6.3)
source(paste0(REPO_PATH, "/simulations/05_run_timing_experiments.R"))

# process results
source(paste0(REPO_PATH, "/simulations/03_process_lm_sims.R"))
source(paste0(REPO_PATH, "/simulations/04_process_probit_sims.R"))
source(paste0(REPO_PATH, "/simulations/06_process_timing_results.R"))

# implementation section 
source(paste0(REPO_PATH, "/simulations/07_implementation.R"))
