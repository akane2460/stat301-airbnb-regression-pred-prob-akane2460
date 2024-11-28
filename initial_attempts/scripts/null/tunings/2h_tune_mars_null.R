# Tuning for MARS null ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(here)
library(doMC)
library(earth)

# Handle conflicts
tidymodels_prefer()

# parallel processing ----
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores - 1)

# load resamples ----
load(here("data/train_folds.rda"))

# load preprocessing/recipe ----
load(here("recipes/null_recipe.rda"))

# set seed
set.seed(21004)

# model specifications ----
mars_spec <- mars(
  num_terms = tune(),
  prod_degree = tune()
) |> 
  set_mode("regression") |> 
  set_engine("earth")

# define workflow ----
mars_wflow <-
  workflow() |> 
  add_model(mars_spec) |> 
  add_recipe(null_recipe)

# hyperparameter tuning values ----
mars_params <- hardhat::extract_parameter_set_dials(mars_spec) |> 
  update(
    num_terms = num_terms(range = c(1L, 5L)),
    prod_degree = prod_degree()
  )

all_params <- bind_rows(mars_params)

# build grid
mars_grid <- grid_regular(all_params, levels = 24)


reg_metrics <- metric_set(mae)

# tune/fit workflow/model ----
null_mars_tuned <-
  tune_grid(mars_wflow,
            resamples = train_folds,
            grid = mars_grid,
            metrics = reg_metrics,
            control = control_grid(save_workflow = TRUE)
  )


# write out results (fitted/trained workflows & runtime info) ----
save(
  null_mars_tuned,
  file = here("initial_attempts/results/null_mars_tuned.rda")
)
