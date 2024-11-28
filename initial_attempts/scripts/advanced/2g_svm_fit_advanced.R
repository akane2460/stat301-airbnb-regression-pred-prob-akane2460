# Tune SVM RBF model

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)
library(stacks)
library(cli)


# Handle common conflicts
tidymodels_prefer()

# parallel processing ----
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores)

# read in the data----
load(here("data/train_regression_cleaned.rda"))

# load the recipes----
load(here("recipes/advanced_recipe.rda"))

# load tuned result----
load(here("initial_attempts/results/advanced_svm_tuned.rda"))

# set seed
set.seed(09991)

# tuned analysis----
# best results
best_results_svm <- select_best(advanced_svm_tuned, metric = "mae")
# best params cost = 0.00543, rbf_sigma = 0.000000206

# model specification ----
svm_spec <-
  svm_rbf(
    cost = 32, 
    rbf_sigma = .00316
  ) |>
  set_mode("regression") |>
  set_engine("kernlab")

# workflow ----
svm_wflow <-
  workflow() |>
  add_model(svm_spec) |>
  add_recipe(advanced_recipe)

# Tuning/fitting ----
advanced_fit_svm <- fit(svm_wflow, train_regression_cleaned)

# Write out results & workflow
save(advanced_fit_svm, file = here("initial_attempts/results/advanced_fit_svm.rda"))