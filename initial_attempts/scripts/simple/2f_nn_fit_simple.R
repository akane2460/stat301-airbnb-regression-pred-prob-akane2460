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
load(here("recipes/simple_recipe.rda"))

# load tuned result----
load(here("initial_attempts/results/simple_nn_tuned.rda"))

# set seed
set.seed(09922)

# tuned analysis----
# best results
best_results_nn <- select_best(simple_nn_tuned, metric = "mae")
# best parameters: hidden_units = 7, penalty = 1

# model specifications----
nn_spec <- 
  mlp(hidden_units = 7, penalty = 1) |> 
  set_mode("regression") |> 
  set_engine("nnet")

# define workflows ----
simple_nn_wflow <- 
  workflow() |> 
  add_model(nn_spec) |> 
  add_recipe(simple_recipe)

# fit workflows/models ----
simple_fit_nn <- fit(simple_nn_wflow, train_regression_cleaned)

# save tune
save(simple_fit_nn, file = here("initial_attempts/results/simple_fit_nn.rda"))
