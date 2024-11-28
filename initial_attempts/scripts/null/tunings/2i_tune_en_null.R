# Define and fit of en null

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

# load training data
load(here("data/train_regression_cleaned.rda"))

# load pre-processing/feature engineering/recipe
load(here("recipes/null_recipe.rda"))

# load folds
load(here("data/train_folds.rda"))

# model specifications ----
en_spec <- 
  linear_reg(penalty = tune(), mixture = tune()) |> 
  set_engine("glmnet") |> 
  set_mode("regression") 

# define workflows ----
# null wflow
null_en_wflow <-
  workflow() |> 
  add_model(en_spec) |> 
  add_recipe(null_recipe)

# hyperparameter tuning values ----
# null hyperparameters
hardhat::extract_parameter_set_dials(null_en_wflow)

null_en_params <- parameters(null_en_wflow) |>  
  update(penalty = penalty(c(0, 1)), mixture = mixture(c(.01, .1)))

null_en_grid <- grid_latin_hypercube(null_en_params, size = 25)

# metric set----
reg_metrics <- metric_set(mae)

# tune workflows/models ----
# null
# set seed
set.seed(129742)
null_en_tuned <- 
  null_en_wflow |> 
  tune_grid(
    train_folds, 
    grid = null_en_grid, 
    metrics = reg_metrics,
    control = control_grid(save_workflow = TRUE)
  )

# write out results (fitted/trained workflows) ----
save(null_en_tuned, file = here("initial_attempts/results/null_en_tuned.rda"))