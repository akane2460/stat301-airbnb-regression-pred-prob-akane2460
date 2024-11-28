# Define and fit nearest neighbor

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
load(here("initial_attempts/results/simple_knn_tuned.rda"))

# set seed
set.seed(0996)

# tuned analysis----

# best results
best_results_knn <- select_best(simple_knn_tuned, metric = "mae")
# performs best with neighbors = 2, dist_power = 2.91

# model specifications----
kknn_spec <- 
  nearest_neighbor(neighbors = 2, dist_power = 2.91) |> 
  set_engine("kknn") |> 
  set_mode("regression")

# define workflows ----
simple_knn_wflow <-
  workflow() |> 
  add_model(kknn_spec) |> 
  add_recipe(simple_recipe)

# fit workflows/models ----
simple_fit_knn <- fit(simple_knn_wflow, train_regression_cleaned)

# save tune
save(simple_fit_knn, file = here("initial_attempts/results/simple_fit_knn.rda"))
