# Define and fit advanced poly

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(here)
library(tictoc)
library(doMC)
library(kernlab)

# handle common conflicts
tidymodels_prefer()

# parallel processing ----
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores - 1)

# read in the data----
load(here("data/train_regression_cleaned.rda"))

# load the recipes----
load(here("recipes/advanced_recipe.rda"))

# load tuned----
load(here("initial_attempts/results/tune_svm_poly_advanced.rda"))

# set seed
set.seed(992741)

# tuned analysis----
# best results
best_results_svm_poly <- select_best(tune_svm_poly_advanced, metric = "mae")
# best parameters: 

# model specifications ----
svm_poly_model <- svm_poly(
  mode = "regression",
  cost = tune(),
  degree = tune(),
  scale_factor = tune()
) |>
  set_mode("regression") |> 
  set_engine("kernlab")

# define workflows ----
svm_poly_wflow <- workflow() |> 
  add_model(svm_poly_model) |> 
  add_recipe(advanced_recipe)

# hyperparameter tuning values ----
svm_poly_params <- hardhat::extract_parameter_set_dials(svm_poly_model)

svm_poly_grid <- grid_latin_hypercube(svm_poly_params, size = 50)

# fit workflow/model ----
# tuning code in here
tune_svm_poly_advanced <- svm_poly_wflow |> 
  tune_grid(
    resamples = train_folds,
    grid = svm_poly_grid,
    control = control_grid(save_workflow = TRUE),
    metrics = metric_set(mae)
  )


# write out results ----
save(tune_svm_poly_advanced,
     file = here("initial_attempts/results/tune_svm_poly_advanced.rda"))