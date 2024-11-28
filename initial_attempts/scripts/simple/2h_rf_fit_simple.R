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
load(here("recipes/simple_recipe.rda"))

# load tuned----
load(here("initial_attempts/results/simple_rf_tuned.rda"))

# set seed
set.seed(457229)

# tuned analysis----
# best results
best_results_rf <- select_best(simple_rf_tuned, metric = "mae")
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
  add_recipe(simple_recipe)

# fit workflows/models ----
simple_fit_rf <- fit(rf_wflow, train_regression_cleaned)

# Write out results & workflow
save(simple_fit_rf , file = here("initial_attempts/results/simple_fit_rf.rda"))