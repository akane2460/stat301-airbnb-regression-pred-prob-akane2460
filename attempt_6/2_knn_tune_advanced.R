## knn tuning for ensemble-- advanced

# load packages ----
library(tidymodels)
library(tidyverse)
library(here)
library(stacks)
library(doMC)

# load data ----
load(here("data/train_folds.rda"))
load(here("recipes/advanced_recipe.rda"))

# handle common conflicts ----
tidymodels_prefer()

# parallel processing ----
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores)

set.seed(6301)

# model specifications ----
kknn_spec <-
  nearest_neighbor(neighbors = tune()) |>
  set_engine("kknn") |>
  set_mode("regression")


# define workflow ----
knn_wflow <- workflow() |> 
  add_model(kknn_spec) |> 
  add_recipe(advanced_recipe)

# tuning parameters ----
set.seed(6301)
knn_params <- extract_parameter_set_dials(kknn_spec) |> 
  update(neighbors = neighbors(range = c(7, 20)))

knn_grid <- grid_latin_hypercube(knn_params, size = 16)

# model tuning ----
set.seed(6301)
knn_tuned_advanced <- tune_grid(
  knn_wflow,
  train_folds,
  grid = knn_grid,
  control = control_stack_grid(),
  metrics = metric_set(mae))

# save results ----
save(knn_tuned_advanced, file = here("attempt_6/results/knn_tuned_advanced.rda"))