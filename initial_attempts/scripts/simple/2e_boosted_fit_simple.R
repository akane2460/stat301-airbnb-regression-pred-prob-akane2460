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
load(here("recipes/simple_recipe.rda"))

# load tuned result----
load(here("initial_attempts/results/simple_boosted_tuned.rda"))

# set seed
set.seed(29174)

# tuned analysis----
# best results
best_results_boosted <- select_best(simple_boosted_tuned, metric = "mae")
# best parameters: mtry = 7, min_n = 21, learn_rate = 1.26

# model specifications----
boosted_spec <- 
  boost_tree(mtry = 7, min_n = 21, learn_rate = 1.26) |> 
  set_engine("xgboost") |> 
  set_mode("regression")

# define workflows ----
simple_boosted_wflow <-
  workflow() |> 
  add_model(boosted_spec) |> 
  add_recipe(simple_recipe)

# fit workflows/models ----
simple_fit_boosted <- fit(simple_boosted_wflow, train_regression_cleaned)

# save tune
save(simple_fit_boosted, file = here("initial_attempts/results/simple_fit_boosted.rda"))
