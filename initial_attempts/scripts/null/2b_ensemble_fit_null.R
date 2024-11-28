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
load(here("initial_attempts/results/null_lm_res.rda"))
# load(here("initial_attempts/results/null_lasso_res.rda"))
# load(here("initial_attempts/results/null_ridge_res.rda"))
load(here("initial_attempts/results/null_knn_res.rda"))
load(here("initial_attempts/results/null_boosted_res.rda"))
load(here("initial_attempts/results/null_nn_res.rda"))
load(here("initial_attempts/results/null_svm_res.rda"))
load(here("initial_attempts/results/null_mars_res.rda"))



# Create data stack ----
null_stack <- stacks() |> 
  add_candidates(null_lm_res) |> 
  # add_candidates(null_lasso_res)
  # add_candidates(null_ridge_res) |> 
  add_candidates(null_knn_res) |> 
  add_candidates(null_boosted_res) |>
  add_candidates(null_nn_res) |>
  add_candidates(null_svm_res) |>
  add_candidates(null_mars_res)

# Fit the stack ----
# penalty values for blending (set penalty argument when blending)
blend_penalty <- c(10^(-6:-1), 0.5, 1, 1.5, 2)

# Blend predictions (tuning step, set seed)
set.seed(912211)

# Save blended model stack for reproducibility & easy reference (for report)
null_model_st <- 
  null_stack |> 
  blend_predictions(penalty = blend_penalty) |> 
  fit_members()

# fit to training set ----
null_model_st <-
  null_model_st |>
  fit_members(test_regression_cleaned)

# Save trained ensemble model for reproducibility & easy reference (for report)
save(null_model_st, file = here("initial_attempts/results/null_model_st.rda"))
