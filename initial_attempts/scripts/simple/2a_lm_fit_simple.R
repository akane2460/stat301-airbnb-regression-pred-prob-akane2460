# Define and fit simple linear regression

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
load(here("recipes/simple_recipe.rda"))

# model specifications ----
lm_spec <- 
  linear_reg() |> 
  set_engine("lm") |> 
  set_mode("regression") 

# define workflows ----
simple_lm_wflow <-
  workflow() |> 
  add_model(lm_spec) |> 
  add_recipe(simple_recipe)

# fit workflows/models ----
simple_fit_lm <- fit(simple_lm_wflow, train_regression_cleaned)

# write out results (fitted/trained workflows) ----
save(simple_fit_lm, file = here("initial_attempts/results/simple_fit_lm.rda"))
