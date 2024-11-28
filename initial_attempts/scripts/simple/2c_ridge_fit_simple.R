# Define and fit of ridge simple

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
load(here("recipes/simple_recipe.rda"))

# model specifications ----
ridge_spec <- 
  linear_reg(penalty = 1, mixture = 0) |> 
  set_engine("glmnet") |> 
  set_mode("regression") 

# define workflows ----
# simple wflow
simple_ridge_wflow <-
  workflow() |> 
  add_model(ridge_spec) |> 
  add_recipe(simple_recipe)

# fit workflows/models ----
simple_fit_ridge <- fit(simple_ridge_wflow, train_regression_cleaned)

# write out results (fitted/trained workflows) ----
save(simple_fit_ridge, file = here("initial_attempts/results/simple_fit_ridge.rda"))
