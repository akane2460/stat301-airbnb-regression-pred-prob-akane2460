## knn tuning for ensemble-- null

# load packages ----
library(tidymodels)
library(tidyverse)
library(here)
library(stacks)
library(doMC)

# load data ----
load(here("data/train_folds.rda"))
load(here("recipes/null_recipe.rda"))

# handle common conflicts ----
tidymodels_prefer()

# parallel processing ----
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores)

set.seed(01298)

# model specifications ----
kknn_spec <-
  nearest_neighbor(neighbors = tune()) |>
  set_engine("kknn") |>
  set_mode("regression")


# define workflow ----
knn_wflow <- workflow() |> 
  add_model(kknn_spec) |> 
  add_recipe(null_recipe)

# tuning parameters ----
set.seed(01298)
knn_params <- extract_parameter_set_dials(kknn_spec) |> 
  update(neighbors = neighbors(range = c(7, 20)))

knn_grid <- grid_latin_hypercube(knn_params, size = 15)

# model tuning ----
set.seed(01298)
knn_tuned_null <- tune_grid(
  knn_wflow,
  train_folds,
  grid = knn_grid,
  control = control_stack_grid(),
  metrics = metric_set(mae))

# save results ----
save(knn_tuned_null, file = here("attempt_5/results/knn_tuned_null.rda"))