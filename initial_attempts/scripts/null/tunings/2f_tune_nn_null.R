# Tune neural networks

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
set.seed(910297)

# model specifications----
nn_spec <- 
  mlp(hidden_units = tune(), penalty = tune()) |> 
  set_mode("regression") |> 
  set_engine("nnet")

# define workflows ----
null_nn_wflow <- workflow() |> 
  add_model(nn_spec) |> 
  add_recipe(null_recipe)

# hyperparameter tuning values ----
# null hyperparameters
hardhat::extract_parameter_set_dials(null_nn_wflow)

null_nn_params <- parameters(null_nn_wflow) |>  
  update(hidden_units = hidden_units(), penalty = penalty())

null_nn_grid <- grid_latin_hypercube(null_nn_params, size = 5)

# tune workflows/models ----

reg_metrics <- metric_set(mae)

# null
# set seed
set.seed(11172128)
null_nn_tuned <- 
  null_nn_wflow |> 
  tune_grid(
    train_folds, 
    grid = null_nn_grid, 
    metrics = reg_metrics,
    control = control_grid(save_workflow = TRUE)
  )

# save tune
save(null_nn_tuned, file = here("initial_attempts/results/null_nn_tuned.rda"))
