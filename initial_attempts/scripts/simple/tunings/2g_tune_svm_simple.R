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
load(here("recipes/simple_recipe.rda"))

# load folds----
load(here("data/train_folds.rda"))

# set seed
set.seed(0973241)

# model specification ----
svm_spec <-
  svm_rbf(
    cost = tune(),
    rbf_sigma = tune()
  ) |>
  set_mode("regression") |>
  set_engine("kernlab")

# # check tuning parameters
# hardhat::extract_parameter_set_dials(svm_spec)

# set-up tuning grid ----
svm_params <- hardhat::extract_parameter_set_dials(svm_spec)

# define grid
svm_grid <- grid_latin_hypercube(svm_params, levels = 20)

# workflow ----
svm_wflow <-
  workflow() |>
  add_model(svm_spec) |>
  add_recipe(simple_recipe)

# Tuning/fitting ----

metric <- metric_set(mae)
ctrl_res <- control_stack_resamples()

simple_svm_tuned <-
  svm_wflow |>
  tune_grid(
    resamples = train_folds,
    grid = svm_grid,
    metrics = metric,
    control = ctrl_res
  )

# Write out results & workflow
save(simple_svm_tuned, file = here("initial_attempts/results/simple_svm_tuned.rda"))
