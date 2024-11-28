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
load(here("recipes/advanced_recipe.rda"))

# load tuned result----
load(here("initial_attempts/results/advanced_mars_tuned.rda"))

# set seed
set.seed(09721)

# tuned analysis----
# best results
best_results_mars <- select_best(advanced_mars_tuned, metric = "mae")
# performs best with 3 terms and 1 prod_degree

# model specifications----
mars_spec <- mars(
  num_terms = 3,
  prod_degree = 1
) |> 
  set_mode("regression") |> 
  set_engine("earth")

# define workflow ----
advanced_mars_wflow <-
  workflow() |> 
  add_model(mars_spec) |> 
  add_recipe(advanced_recipe)

# fit workflows/models ----
advanced_fit_mars <- fit(advanced_mars_wflow, train_regression_cleaned)

# write out results (fitted/trained workflows & runtime info) ----
save(
  advanced_fit_mars,
  file = here("initial_attempts/results/advanced_fit_mars.rda")
)