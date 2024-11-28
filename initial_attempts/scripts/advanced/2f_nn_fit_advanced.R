# Define and fit nn

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

# read in the data----
load(here("data/train_regression_cleaned.rda"))

# load the recipes----
load(here("recipes/advanced_recipe.rda"))

# load tuned result----
load(here("initial_attempts/results/advanced_nn_tuned.rda"))

# set seed
set.seed(09922)

# tuned analysis----
# best results
best_results_nn <- select_best(advanced_nn_tuned, metric = "mae")
# best parameters: hidden_units = 7, penalty = 1

# model specifications----
nn_spec <- 
  mlp(hidden_units = 7, penalty = 1) |> 
  set_mode("regression") |> 
  set_engine("nnet")

# define workflows ----
advanced_nn_wflow <- 
  workflow() |> 
  add_model(nn_spec) |> 
  add_recipe(advanced_recipe)

# fit workflows/models ----
advanced_fit_nn <- fit(advanced_nn_wflow, train_regression_cleaned)

# save tune
save(advanced_fit_nn, file = here("initial_attempts/results/advanced_fit_nn.rda"))