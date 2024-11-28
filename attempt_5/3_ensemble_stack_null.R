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
load(here("attempt_5/results/boosted_tuned_null.rda"))
load(here("attempt_5/results/knn_tuned_null.rda"))

set.seed(0128)

# Create data stack ----
null_stack <- stacks() |> 
  add_candidates(boosted_tuned_null) |> 
  add_candidates(knn_tuned_null) 
  # select(-boosted_tuned_null_1_10) # excluding the column with missing predictions
          # unclear as to why they arose, potentially due to similar predictions
          # seen in the knn?

# account for NAs 
cols_with_na <- names(null_stack)[apply(is.na(null_stack), 2, any)]

null_stack <- null_stack |> 
  select(-cols_with_na)

# is.na(null_stack)

# fit the stack ----
# penalty values for blending (set penalty argument when blending)
blend_penalty <- c(10^(-6:-1), 0.5, 1, 1.5, 2)

# blend predictions
set.seed(641)

# save blended model stack null
null_data_stack <- 
  null_stack |> 
  blend_predictions(penalty = blend_penalty)

# save stack blended for analysis 
save(null_data_stack, file = here("attempt_5/results/null_data_stack.rda"))

# fit to training set ----
ensemble_model_null <- null_data_stack |> 
  fit_members()

# Save trained ensemble model for reproducibility & easy reference (for report)
save(ensemble_model_null, file = here("attempt_5/results/ensemble_model_null.rda"))

