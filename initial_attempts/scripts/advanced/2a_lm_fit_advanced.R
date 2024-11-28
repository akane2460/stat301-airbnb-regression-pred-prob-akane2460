# Define and fit advanced linear regression

# worse performing lm

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)

# handle common conflicts
tidymodels_prefer()

# parallel processing
num_cores <- parallel::detectCores(logical = TRUE)

registerDoMC(cores = num_cores)

# load training data----
load(here("data/train_regression_cleaned.rda"))

# load the recipes----
load(here("recipes/advanced_recipe.rda"))

# model specifications ----
lm_spec <- 
  linear_reg() |> 
  set_engine("lm") |> 
  set_mode("regression") 

# define workflows ----
advanced_lm_wflow <-
  workflow() |> 
  add_model(lm_spec) |> 
  add_recipe(advanced_recipe)

# fit workflows/models ----
advanced_fit_lm <- fit(advanced_lm_wflow, train_regression_cleaned)

# write out initial_attempts/results (fitted/trained workflows) ----
save(advanced_fit_lm, file = here("initial_attempts/results/advanced_fit_lm.rda"))