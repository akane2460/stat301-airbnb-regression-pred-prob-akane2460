# Define and fit mars

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
load(here("initial_attempts/results/null_mars_tuned.rda"))

# set seed
set.seed(0922221)

# tuned analysis----
# best results
best_results_mars <- select_best(null_mars_tuned, metric = "mae")
# performs best with 5 terms and 2 prod_degree

# model specifications----
# mars_spec <- mars(
#   num_terms = 5,
#   prod_degree = 2
# ) |> 
#   set_mode("regression") |> 
#   set_engine("earth")
# 
# # define workflow ----
# null_mars_wflow <-
#   workflow() |> 
#   add_model(mars_spec) |> 
#   add_recipe(null_recipe)
# 
# # fit workflows/models ----
# null_fit_mars <- fit(null_mars_wflow, train_regression_cleaned)
# 
# # write out results (fitted/trained workflows & runtime info) ----
# save(
#   null_fit_mars,
#   file = here("initial_attempts/results/null_fit_mars.rda")
# )