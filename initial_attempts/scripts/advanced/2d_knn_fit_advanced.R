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
load(here("recipes/advanced_recipe.rda"))

# load tuned result----
load(here("initial_attempts/results/advanced_knn_tuned.rda"))

# set seed
set.seed(02124)

# tuned analysis----

# best results
best_results_knn <- select_best(advanced_knn_tuned, metric = "mae")
# performs best with neighbors = 2, dist_power = 2.91

# model specifications----
kknn_spec <- 
  nearest_neighbor(neighbors = 2, dist_power = 2.91) |> 
  set_engine("kknn") |> 
  set_mode("regression")

# define workflows ----
advanced_knn_wflow <-
  workflow() |> 
  add_model(kknn_spec) |> 
  add_recipe(advanced_recipe)

# fit workflows/models ----
advanced_fit_knn <- fit(advanced_knn_wflow, train_regression_cleaned)

# save tune
save(advanced_fit_knn, file = here("initial_attempts/results/advanced_fit_knn.rda"))