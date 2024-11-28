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
load(here("attempt_6/results/boosted_tuned_advanced.rda"))
load(here("attempt_6/results/knn_tuned_advanced.rda"))

set.seed(0128)

# Create data stack ----
advanced_stack <- stacks() |> 
  add_candidates(boosted_tuned_advanced) |> 
  add_candidates(knn_tuned_advanced) 
  # select(-boosted_tuned_advanced_1_10) # excluding the column with missing predictions
          # unclear as to why they arose, potentially due to similar predictions
          # seen in the knn?

# account for NAs 
cols_with_na <- names(advanced_stack)[apply(is.na(advanced_stack), 2, any)]

advanced_stack <- advanced_stack |> 
  select(-cols_with_na)

# is.na(advanced_stack)

# fit the stack ----
# penalty values for blending (set penalty argument when blending)
blend_penalty <- c(10^(-6:-1), 0.5, 1, 1.5, 2)

# blend predictions
set.seed(641)

# save blended model stack advanced
advanced_data_stack <- 
  advanced_stack |> 
  blend_predictions(penalty = blend_penalty)

# save stack blended for analysis 
save(advanced_data_stack, file = here("attempt_6/results/advanced_data_stack.rda"))

# fit to training set ----
ensemble_model_advanced <- advanced_data_stack |> 
  fit_members()

# Save trained ensemble model for reproducibility & easy reference (for report)
save(ensemble_model_advanced, file = here("attempt_6/results/ensemble_model_advanced.rda"))
