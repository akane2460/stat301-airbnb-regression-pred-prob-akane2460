# Train & explore ensemble model

# Load package(s) ----
library(tidymodels)
library(tidyverse)
library(here)
library(stacks)
library(cli)

# Handle common conflicts
tidymodels_prefer()

# load models----
load(here("data/test_regression_cleaned.rda"))

# Load candidate model info ----
load(here("attempt_1_not_selected/results/null_lm_res.rda"))
load(here("attempt_1_not_selected/results/null_knn_res.rda"))
load(here("attempt_1_not_selected/results/null_boosted_res.rda"))
load(here("attempt_1_not_selected/results/null_nn_res.rda"))
load(here("attempt_1_not_selected/results/null_svm_res.rda"))
load(here("attempt_1_not_selected/results/null_mars_res.rda"))

# set seed
set.seed(912211)

# Create data stack ----
null_stack <- stacks() |> 
  add_candidates(null_lm_res) |> 
  add_candidates(null_knn_res) |> 
  add_candidates(null_boosted_res) |>
  add_candidates(null_nn_res) |>
  add_candidates(null_svm_res) |>
  add_candidates(null_mars_res)

# Fit the stack ----
# penalty values for blending (set penalty argument when blending)
blend_penalty <- c(10^(-6:-1), 0.5, 1, 1.5, 2)

set.seed(912211)
# Save blended model stack for reproducibility & easy reference (for report)
null_model_st <- 
  null_stack |> 
  blend_predictions(penalty = blend_penalty) 

set.seed(912211)

autoplot(null_model_st) +
  theme_minimal()

ensemble_contrib_plot <- autoplot(null_data_stack, type = "weights") +
  theme_minimal() 

ggsave(filename = here("plots/ensemble_contrib_plot.png"), ensemble_contrib_plot)

# fit to training set ----
null_model_st <-
  null_model_st |>
  fit_members()

# Save trained ensemble model for reproducibility & easy reference (for report)
save(null_model_st, file = here("attempt_1_not_selected/results/null_model_st.rda"))
