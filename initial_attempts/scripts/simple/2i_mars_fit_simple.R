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
load(here("recipes/simple_recipe.rda"))

# load tuned result----
load(here("initial_attempts/results/simple_mars_tuned.rda"))

# set seed
set.seed(0922221)

# tuned analysis----
# best results
best_results_mars <- select_best(simple_mars_tuned, metric = "mae")
# performs best with 3 terms and 1 prod_degree

# model specifications----
mars_spec <- mars(
  num_terms = 3,
  prod_degree = 1
) |> 
  set_mode("regression") |> 
  set_engine("earth")

# define workflow ----
simple_mars_wflow <-
  workflow() |> 
  add_model(mars_spec) |> 
  add_recipe(simple_recipe)

# fit workflows/models ----
simple_fit_mars <- fit(simple_mars_wflow, train_regression_cleaned)

# write out results (fitted/trained workflows & runtime info) ----
save(
  simple_fit_mars,
  file = here("initial_attempts/results/simple_fit_mars.rda")
)