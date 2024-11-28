# Define and fit of ridge advanced

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
load(here("recipes/advanced_recipe.rda"))

# model specifications ----
ridge_spec <- 
  linear_reg(penalty = 1, mixture = 0) |> 
  set_engine("glmnet") |> 
  set_mode("regression") 

# define workflows ----
# advanced wflow
advanced_ridge_wflow <-
  workflow() |> 
  add_model(ridge_spec) |> 
  add_recipe(advanced_recipe)

# fit workflows/models ----
advanced_fit_ridge <- fit(advanced_ridge_wflow, train_regression_cleaned)

# write out initial_attempts/results (fitted/trained workflows) ----
save(advanced_fit_ridge, file = here("initial_attempts/results/advanced_fit_ridge.rda"))

# write out initial_attempts/results (fitted/trained workflows) ----
save(advanced_fit_ridge, file = here("initial_attempts/results/advanced_fit_ridge.rda"))