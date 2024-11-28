# Tune random forest model

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)
library(stacks)
library(cli)
library(ranger)

# Handle common conflicts
tidymodels_prefer()

# parallel processing ----
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores)

# read in the data----
load(here("data/train_regression_cleaned.rda"))

# load the recipes----
load(here("recipes/null_recipe.rda"))

# load folds----
load(here("data/train_folds.rda"))

# set seed
set.seed(4236)

# model specification ----
rf_spec <- 
  rand_forest(trees = tune(), min_n = tune(), mtry = tune()) |> 
  set_engine("ranger") |> 
  set_mode("regression")

# define workflows ----
rf_wflow <-
  workflow() |> 
  add_model(rf_spec) |> 
  add_recipe(null_recipe)

# hyperparameter tuning values ----
# check ranges for hyperparameters
hardhat::extract_parameter_set_dials(rf_wflow)

# change hyperparameter ranges
rf_params <- parameters(rf_wflow) |> 
  update(mtry = mtry(c(1, 7)),
         min_n = min_n(),
         trees = trees()) 

# build tuning grid
rf_grid <- grid_regular(rf_params, levels = 5)

# fit workflows/models ----

reg_metrics <- metric_set(mae)

# set seed
set.seed(012297)
null_rf_tuned <- 
  rf_wflow |> 
  tune_grid(
    train_folds, 
    grid = rf_grid, 
    metrics = reg_metrics,
    control = control_grid(save_workflow = TRUE)
  )

# Write out results & workflow
save(null_rf_tuned, file = here("initial_attempts/results/null_rf_tuned.rda"))