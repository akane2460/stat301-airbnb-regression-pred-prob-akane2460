# Tune random forest model

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)
library(stacks)
library(cli)
library(ranger)

# Handle common conflicts
tidymodels_prefer()

# parallel processing ----
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores)

# read in the data----
load(here("data/train_regression_cleaned.rda"))

# load the recipes----
load(here("recipes/advanced_recipe.rda"))

# load tuned----
load(here("initial_attempts/results/advanced_rf_tuned.rda"))

# set seed
set.seed(41112)

# tuned analysis----
# best results
best_results_rf <- select_best(advanced_rf_tuned, metric = "mae")
# best parameters: trees = 1, min_n = 11, mtry = 7

# model specification ----
rf_spec <- 
  rand_forest(trees = 1, min_n = 11, mtry = 7) |> 
  set_engine("ranger") |> 
  set_mode("regression")

# define workflows ----
rf_wflow <-
  workflow() |> 
  add_model(rf_spec) |> 
  add_recipe(advanced_recipe)

# fit workflows/models ----
advanced_fit_rf <- fit(rf_wflow, train_regression_cleaned)

# Write out results & workflow
save(advanced_fit_rf , file = here("initial_attempts/results/advanced_fit_rf.rda"))