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
load(here("recipes/simple_recipe.rda"))

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
simple_nn_wflow <- workflow() |> 
  add_model(nn_spec) |> 
  add_recipe(simple_recipe)

# hyperparameter tuning values ----
# simple hyperparameters
hardhat::extract_parameter_set_dials(simple_nn_wflow)

simple_nn_params <- parameters(simple_nn_wflow) |>  
  update(hidden_units = hidden_units(), penalty = penalty())

simple_nn_grid <- grid_regular(simple_nn_params, levels = 5)

# tune workflows/models ----

reg_metrics <- metric_set(mae)

# simple
# set seed
set.seed(67543768)
simple_nn_tuned <- 
  simple_nn_wflow |> 
  tune_grid(
    train_folds, 
    grid = simple_nn_grid, 
    metrics = reg_metrics,
    control = control_grid(save_workflow = TRUE)
  )

# save tune
save(simple_nn_tuned, file = here("initial_attempts/results/simple_nn_tuned.rda"))
