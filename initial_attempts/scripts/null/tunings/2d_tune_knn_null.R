# Tune K nearest neighbor null

# random processes present

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)
library(parallel)

# handle common conflicts
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
set.seed(2864312)

# model specifications----
kknn_spec <- 
  nearest_neighbor(neighbors = tune(),
                   dist_power = tune()) |> 
  set_engine("kknn") |> 
  set_mode("regression")

# define workflows ----
null_knn_wflow <-
  workflow() |> 
  add_model(kknn_spec) |> 
  add_recipe(null_recipe)

# hyperparameter tuning values ----
# null hyperparameters
hardhat::extract_parameter_set_dials(null_knn_wflow)

null_knn_params <- parameters(null_knn_wflow) |>  
  update(neighbors = neighbors(c(1,11)),
         dist_power = dist_power(range = c(1, 3)))

null_knn_grid <- grid_latin_hypercube(null_knn_params, size = 10)

# metric set
reg_metrics <- metric_set(mae)

# tune workflows/models ----
# null
# set seed
set.seed(2864312)
null_knn_tuned <- 
  null_knn_wflow |> 
  tune_grid(
    train_folds, 
    grid = null_knn_grid, 
    metrics = reg_metrics,
    control = control_grid(save_workflow = TRUE)
  )

# save tune
save(null_knn_tuned, file = here("initial_attempts/results/null_knn_tuned.rda"))