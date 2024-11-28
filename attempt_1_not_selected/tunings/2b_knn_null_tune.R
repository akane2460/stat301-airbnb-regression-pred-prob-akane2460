# Ensemble model contributors

# Load package(s) ----
library(tidymodels)
library(tidyverse)
library(here)
library(stacks)
library(cli)
library(doMC)

# Handle common conflicts
tidymodels_prefer()

# parallel processing
num_cores <- parallel::detectCores(logical = TRUE)

registerDoMC(cores = num_cores)

# read in the data----
load(here("data/train_regression_cleaned.rda"))

# load the recipes----
load(here("recipes/null_recipe.rda"))

# load folds----
load(here("data/train_folds.rda"))

# set seed
set.seed(265421)

# model specification ----
knn_spec <-
  nearest_neighbor(
    neighbors = tune(),
    dist_power = tune()
  ) |>
  set_mode("regression") |>
  set_engine("kknn")

# set seed
set.seed(265421)
# set-up tuning grid ----
knn_params <- hardhat::extract_parameter_set_dials(knn_spec) |>
  update(neighbors = neighbors(range = c(1, 20)),
         dist_power = dist_power(range = c(0, 2)
                                 )
         )

# set seed
set.seed(265421)
# define grid
knn_grid <- grid_regular(knn_params, levels = 5)

# workflow ----
knn_wflow <-
  workflow() |>
  add_model(knn_spec) |>
  add_recipe(null_recipe)

# Tuning/fitting ----
set.seed(265421)

metric <- metric_set(mae)
ctrl_res <- control_stack_resamples()

knn_res <-
  knn_wflow |>
  tune_grid(
    resamples = train_folds,
    grid = knn_grid,
    metrics = metric,
    control = ctrl_res
  )

# Write out results & workflow
save(knn_res, file = here("results/knn_res.rda"))

# Write out results & workflow
save(null_knn_res, file = here("attempt_1/results/null_knn_res.rda"))