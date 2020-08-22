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
REPO_PATH = "~/fastbayes"
results_path = paste0(REPO_PATH, "/simulations/results/")

source(paste0(REPO_PATH, "/simulations/sim_helpers.R"))
source(paste0(REPO_PATH, "/simulations/sim_helpers_lm.R"))
source(paste0(REPO_PATH, "/simulations/sim_helpers_probit.R"))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# process simulation result csv file
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
timing_results_path = paste0(results_path, "timing_results.csv")
results_df <- read_csv(timing_results_path)

results_summary <-
  results_df %>%
  group_by(model, algo, n, p, fix) %>%
  summarise(
    mean = mean(time),
    median = median(time)
  ) %>%
  mutate(
    model = factor(
      model,
      levels = c("linear", "binom"),
      labels = c("Normal", "Binary")
    ),
    corr = factor(
      str_split(algo, "_")[[1]][2],
      levels = c("0", "1"),
      labels = c("Corr", "Indep")
    ),
    algo = factor(
      str_split(algo, "_")[[1]][1],
      levels = c("glmnet", "cavi", "svi"),
      labels = c("GLM", "CAVI", "SVI")
    ),
    model_type = paste0(algo, "-", corr),
    model_type = ifelse(model_type == "GLM-NA", "GLM", model_type)
  )

p1 <- results_summary %>%
  ungroup() %>%
  filter(fix == "N_fix") %>%
  select(p, mean, median, model_type, algo, n, model) %>%
  pivot_longer(c("mean", "median")) %>%
  ggplot(aes(x = p, y = value, color = model_type)) +
  geom_line(size = 1, aes(linetype = name)) +
  facet_grid( ~ model)

figs_path <- paste0(REPO_PATH, "/paper/figs/")
ggsave(
  p1, filename = paste0(figs_path, "timing_n_fix.pdf"), device = "pdf",
  width = 10, height = 7
)

p2 <- results_summary %>%
  ungroup() %>%
  filter(fix == "P_fix") %>%
  select(p, mean, median, model_type, algo, n, model) %>%
  pivot_longer(c("mean", "median")) %>%
  ggplot(aes(x = n, y = value, color = model_type)) +
  geom_line(size = 1, aes(linetype = name)) +
  facet_grid(model ~ ., scales = 'free')

ggsave(
  p2, filename = paste0(figs_path, "timing_p_fix.pdf"), device = "pdf",
  width = 10, height = 7
)
