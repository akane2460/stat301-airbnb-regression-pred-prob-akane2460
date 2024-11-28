# Define and fit of lasso advanced

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
load(here("recipes/advanced_recipe.rda"))

# model specifications ----
lasso_spec <- 
  linear_reg(penalty = 0, mixture = 1) |> 
  set_engine("glmnet") |> 
  set_mode("regression") 

# define workflows ----
# advanced wflow
advanced_lasso_wflow <-
  workflow() |> 
  add_model(lasso_spec) |> 
  add_recipe(advanced_recipe)

# fit workflows/models ----
advanced_fit_lasso <- fit(advanced_lasso_wflow, train_regression_cleaned)

# write out initial_attempts/results (fitted/trained workflows) ----
save(advanced_fit_lasso, file = here("initial_attempts/results/advanced_fit_lasso.rda"))