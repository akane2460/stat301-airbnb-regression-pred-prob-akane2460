# Tune boosted tree simple

# random processes present

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)
library(xgboost)
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
boosted_spec <- 
  boost_tree(mtry = tune(), min_n = tune(), learn_rate = tune()) |> 
  set_engine("xgboost") |> 
  set_mode("regression")

# define workflows ----
simple_boosted_wflow <-
  workflow() |> 
  add_model(boosted_spec) |> 
  add_recipe(simple_recipe)

# hyperparameter tuning values ----
# simple hyperparameters
hardhat::extract_parameter_set_dials(simple_boosted_wflow)

simple_boosted_params <- parameters(simple_boosted_wflow) |>  
  update(mtry = mtry(c(1, 13)), learn_rate = learn_rate(c(.01, .1)))

simple_boosted_grid <- grid_regular(simple_boosted_params, levels = 5)

# metric set----
reg_metrics <- metric_set(mae)

# tune workflows/models ----
# simple
# set seed
set.seed(02973)
simple_boosted_tuned <- 
  simple_boosted_wflow |> 
  tune_grid(
    train_folds, 
    grid = simple_boosted_grid, 
    metrics = reg_metrics,
    control = control_grid(save_workflow = TRUE)
  )

# save tune
save(simple_boosted_tuned, file = here("initial_attempts/results/simple_boosted_tuned.rda"))