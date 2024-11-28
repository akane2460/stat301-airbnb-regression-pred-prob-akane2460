# define and fit elastic net 

# random processes present

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(parallel)
library(doMC)

# handle common conflicts
tidymodels_prefer()

# parallel processing 
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores)

# load training data
load(here("data/train_regression_cleaned.rda"))

# load resamples ----
load(here("data/train_folds.rda"))

# load pre-processing/feature engineering/recipe
load(here("recipes/simple_recipe.rda"))

# model specifications ----
elastic_net_spec <- 
  linear_reg(penalty = tune(), mixture = tune()) |> 
  set_engine("glmnet", family = "binomial") |> 
  set_mode("regression") 

# define workflows ----
# simple  wflow
simple_en_wflow <-
  workflow() |> 
  add_model(elastic_net_spec) |> 
  add_recipe(simple_recipe)

# hyperparameter tuning values ----
# simple hyperparameters
hardhat::extract_parameter_set_dials(simple_en_wflow)

simple_en_params <- parameters(simple_en_wflow) |>  
  update(penalty = penalty(c(0, 1)), mixture = mixture(c(0, 1)))

simple_en_grid <- grid_regular(simple_en_params, levels = 5)

# tune workflows/models ----
# simple tune
set.seed(109274)
simple_en_tuned <- 
  simple_en_wflow |> 
  tune_grid(
    train_folds, 
    grid = simple_en_grid, 
    control = control_grid(save_workflow = TRUE)
  )


# write out tuned results (fitted/trained workflows) ----
save(simple_en_tuned, file = here("initial_attempts/results/simple_en_tuned.rda"))

# load(here("initial_attempts/results/simple_en_tuned.rda"))
# 
# # selecting best parameters
# en_model_set <- as_workflow_set(
#   en_simple = simple_en_tuned
# )
# 
# en_accuracy_metrics <- en_model_set |>
#   collect_metrics() |>
#   filter(.metric == "roc_auc")
# 
# en_max_accuracy <- en_accuracy_metrics |>
#   group_by(wflow_id) |>
#   slice_max(mean) |>
#   distinct(wflow_id, .keep_all = TRUE)
# 
# best_results_en <- select_best(simple_en_tuned, metric = "roc_auc")
# performs best when penalty = 1 and mixture = 0, so not worth exploring further