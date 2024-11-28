# Define and fit nearest neighbor

# BEST PERFORMING KNN

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
# load(here("initial_attempts/results/null_knn_tuned.rda"))

# set seed
set.seed(0912796)

# tuned analysis----

# # best results
# best_results_knn <- select_best(null_knn_tuned, metric = "mae")
# # performs best with neighbors = 7, dist_power = 1.12

# model specifications----
kknn_spec <-
  nearest_neighbor(neighbors = 7, dist_power = 1.12) |>
  set_engine("kknn") |>
  set_mode("regression")

# define workflows ----
null_knn_wflow <-
  workflow() |>
  add_model(kknn_spec) |>
  add_recipe(null_recipe)

# fit workflows/models ----
null_fit_knn <- fit(null_knn_wflow, train_regression_cleaned)

# save tune
save(null_fit_knn, file = here("attempt_3_not_selected/results/null_fit_knn.rda"))
