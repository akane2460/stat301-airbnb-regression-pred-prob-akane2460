## boosted tuning for ensemble-- advanced

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

set.seed(8453)

# parallel processing ----
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores)

# model specifications ----
boosted_spec <- boost_tree(mtry = tune(),
                      min_n = tune(), learn_rate = tune(), trees = 1000) |> 
  set_engine("xgboost") |> 
  set_mode("regression")

# define workflow ----
boosted_wflow <- workflow() |> 
  add_model(boosted_spec) |> 
  add_recipe(advanced_recipe)

# tuning parameters ----
set.seed(8453)
boosted_params <- extract_parameter_set_dials(boosted_spec) |> 
  update(mtry = mtry(range = c(11, 15)),
         min_n = min_n(range = c(5, 9)),
         learn_rate = learn_rate(range = c(-2, 2)))

boosted_grid <- grid_latin_hypercube(boosted_params, size = 20)

# model tuning ----
set.seed(8453)
boosted_tuned_advanced <- tune_grid(
  boosted_wflow,
  train_folds,
  grid = boosted_grid,
  control = control_stack_grid(),
  metrics = metric_set(mae))

# save results ----
save(boosted_tuned_advanced, file = here("attempt_6/results/boosted_tuned_advanced.rda"))
