# Define and fit of lasso null

# random processes present

# worse performing

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
lasso_spec <- 
  linear_reg(penalty = 0, mixture = 1) |> 
  set_engine("glmnet") |> 
  set_mode("regression") 

# define workflows ----
# null wflow
null_lasso_wflow <-
  workflow() |> 
  add_model(lasso_spec) |> 
  add_recipe(null_recipe)

# fit workflows/models ----
null_fit_lasso <- fit(null_lasso_wflow, train_regression_cleaned)

# write out results (fitted/trained workflows) ----
save(null_fit_lasso, file = here("initial_attempts/results/null_fit_lasso.rda"))