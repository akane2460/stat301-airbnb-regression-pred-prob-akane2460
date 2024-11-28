# Define and fit of ridge null

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

# model specifications ----
ridge_spec <- 
  linear_reg(penalty = 1, mixture = 0) |> 
  set_engine("glmnet") |> 
  set_mode("regression") 

# define workflows ----
# null wflow
null_ridge_wflow <-
  workflow() |> 
  add_model(ridge_spec) |> 
  add_recipe(null_recipe)

# fit workflows/models ----
null_fit_ridge <- fit(null_ridge_wflow, train_regression_cleaned)

# write out results (fitted/trained workflows) ----
save(null_fit_ridge, file = here("initial_attempts/results/null_fit_ridge.rda"))

