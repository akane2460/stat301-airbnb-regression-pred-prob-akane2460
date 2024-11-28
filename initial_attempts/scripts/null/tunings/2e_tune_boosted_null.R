# Tune boosted tree null

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
load(here("recipes/null_recipe.rda"))

# load folds----
load(here("data/train_folds.rda"))

# set seed
set.seed(0292222222)

# model specifications----
boosted_spec <- 
  boost_tree(mtry = tune(), min_n = tune(), learn_rate = tune()) |> 
  set_engine("xgboost") |> 
  set_mode("regression")

# define workflows ----
null_boosted_wflow <-
  workflow() |> 
  add_model(boosted_spec) |> 
  add_recipe(null_recipe)

# hyperparameter tuning values ----
# null hyperparameters
hardhat::extract_parameter_set_dials(null_boosted_wflow)

null_boosted_params <- parameters(null_boosted_wflow) |>  
  update(mtry = mtry(c(1, 13)), learn_rate = learn_rate(c(.01, .1)))

null_boosted_grid <- grid_regular(null_boosted_params, levels = 5)

# metric set----
reg_metrics <- metric_set(mae)

# tune workflows/models ----
# null
# set seed
set.seed(0292222222)
null_boosted_tuned <- 
  null_boosted_wflow |> 
  tune_grid(
    train_folds, 
    grid = null_boosted_grid, 
    metrics = reg_metrics,
    control = control_grid(save_workflow = TRUE)
  )

# save tune
save(null_boosted_tuned, file = here("initial_attempts/results/null_boosted_tuned.rda"))