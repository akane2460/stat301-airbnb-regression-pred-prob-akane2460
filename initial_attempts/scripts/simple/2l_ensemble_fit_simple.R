# Train & explore ensemble model

# Load package(s) ----
library(tidymodels)
library(tidyverse)
library(here)
library(stacks)
library(cli)

# Handle common conflicts
tidymodels_prefer()

# Load candidate model info ----
load(here("initial_attempts/results/simple_lm_res.rda"))
load(here("initial_attempts/results/simple_lasso_res.rda"))
load(here("initial_attempts/results/simple_ridge_res.rda"))
load(here("initial_attempts/results/simple_knn_res.rda"))
load(here("initial_attempts/results/simple_boosted_res.rda"))
load(here("initial_attempts/results/simple_nn_res.rda"))
load(here("initial_attempts/results/simple_svm_res.rda"))
# load(here("initial_attempts/results/simple_rf_res.rda"))
load(here("initial_attempts/results/simple_mars_res.rda"))

# Create data stack ----
simple_stack <- stacks() |> 
  add_candidates(simple_lm_res) |> 
  add_candidates(simple_lasso_res) |> 
  add_candidates(simple_ridge_res) |> 
  add_candidates(simple_knn_res) |>
  add_candidates(simple_boosted_res) |> 
  add_candidates(simple_nn_res) |> 
  add_candidates(simple_svm_res) |> 
  # add_candidates(simple_rf_res) |>
  add_candidates(simple_mars_res) 


# Fit the stack ----
# penalty values for blending (set penalty argument when blending)
blend_penalty <- c(10^(-6:-1), 0.5, 1, 1.5, 2)

# Blend predictions (tuning step, set seed)
set.seed(912211)

# Save blended model stack for reproducibility & easy reference (for report)
simple_model_st <- 
  simple_stack |> 
  blend_predictions(penalty = blend_penalty) |> 
  fit_members()

# fit to training set ----
# simple_model_st <-
#   simple_model_st |>
#   fit_members(test_regression_cleaned)

# Save trained ensemble model for reproducibility & easy reference (for report)
save(simple_model_st, file = here("initial_attempts/results/simple_model_st.rda"))
