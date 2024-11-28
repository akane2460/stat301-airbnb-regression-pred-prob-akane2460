# Tune K nearest neighbor

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
load(here("recipes/simple_recipe.rda"))

# load folds----
load(here("data/train_folds.rda"))

# model specifications----
kknn_spec <- 
  nearest_neighbor(neighbors = tune(),
                   dist_power = tune()) |> 
  set_engine("kknn") |> 
  set_mode("regression")

# define workflows ----
simple_knn_wflow <-
  workflow() |> 
  add_model(kknn_spec) |> 
  add_recipe(simple_recipe)

# hyperparameter tuning values ----
# simple hyperparameters
hardhat::extract_parameter_set_dials(simple_knn_wflow)

simple_knn_params <- parameters(simple_knn_wflow) |>  
  update(neighbors = neighbors(c(1,11)),
         dist_power = dist_power(range = c(1, 3)))

simple_knn_grid <- grid_latin_hypercube(simple_knn_params, size = 10)

# metric set
reg_metrics <- metric_set(mae)

# tune workflows/models ----
# simple
# set seed
set.seed(22112)
simple_knn_tuned <- 
  simple_knn_wflow |> 
  tune_grid(
    train_folds, 
    grid = simple_knn_grid, 
    metrics = reg_metrics,
    control = control_grid(save_workflow = TRUE)
  )

# save tune
save(simple_knn_tuned, file = here("initial_attempts/results/simple_knn_tuned.rda"))