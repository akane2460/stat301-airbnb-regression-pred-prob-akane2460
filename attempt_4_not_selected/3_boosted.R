# Define and fit boosted

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
# load(here("initial_attempts/results/null_boosted_tuned.rda"))

# set seed
set.seed(29174)

# tuned analysis----
# best results
# best_results_boosted <- select_best(null_boosted_tuned, metric = "mae")
# best parameters: mtry = 13, min_n = 40, learn_rate = 1.02

# model specifications----
boosted_spec <-
  boost_tree(mtry = 13, min_n = 40, learn_rate = 1.02) |>
  set_engine("xgboost") |>
  set_mode("regression")

# define workflows ----
null_boosted_wflow <-
  workflow() |>
  add_model(boosted_spec) |>
  add_recipe(null_recipe)

# fit workflows/models ----
null_fit_boosted <- fit(null_boosted_wflow, train_regression_cleaned)

# save tune
save(null_fit_boosted, file = here("attempt_4_not_selected/results/null_fit_boosted.rda"))