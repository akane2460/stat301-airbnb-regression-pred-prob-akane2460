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
load(here("recipes/null_recipe.rda"))

# load tuned result----
load(here("initial_attempts/results/null_nn_tuned.rda"))

# set seed
set.seed(0918762)

# tuned analysis----
# best results
best_results_nn <- select_best(null_nn_tuned, metric = "mae")
# best parameters: hidden_units = 9, penalty = .00909

# model specifications----
# nn_spec <-
#   mlp(hidden_units = 1, penalty = 0.00000681) |>
#   set_mode("regression") |>
#   set_engine("nnet")
# 
# # define workflows ----
# null_nn_wflow <-
#   workflow() |>
#   add_model(nn_spec) |>
#   add_recipe(null_recipe)
# 
# # fit workflows/models ----
# null_fit_nn <- fit(null_nn_wflow, train_regression_cleaned)
# 
# # save tune
# save(null_fit_nn, file = here("initial_attempts/results/null_fit_nn.rda"))
