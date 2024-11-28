# Analysis Simple

## Load Packages ----
library(tidyverse)
library(tidymodels)
library(skimr)
library(janitor)
library(here)
library(knitr)
library(corrplot)
library(stacks)

tidymodels_prefer()

# read in the data----
load(here("data/train_regression_cleaned.rda"))
load(here("data/test_regression_cleaned.rda"))

# simple log reg----
load(here("initial_attempts/results/simple_fit_lm.rda"))

simple_lm_submission <- bind_cols(test_regression_cleaned,
                                       predict(simple_fit_lm, test_regression_cleaned)) |>
  select(id, .pred) |>
  rename(predicted = .pred)

simple_lm_submission <- simple_lm_submission |>
  mutate(
    predicted = 10^predicted
  )

write_csv(simple_lm_submission, file = here("initial_attempts/submissions/simple_lm_submission.csv"))


# simple lasso----
load(here("initial_attempts/results/simple_fit_lasso.rda"))

simple_fit_lasso_submission <- bind_cols(test_regression_cleaned,
                                       predict(simple_fit_lasso, test_regression_cleaned)) |>
  select(id, .pred) |>
  rename(predicted = .pred)

simple_fit_lasso_submission <- simple_fit_lasso_submission |>
  mutate(
    predicted = VGAM::yeo.johnson(predicted, lambda = -0.172926, inverse = TRUE) # see recipes script for lambda work
  )

write_csv(simple_fit_lasso_submission, file = here("initial_attempts/submissions/simple_fit_lasso_submission.csv"))

# simple ridge----
load(here("initial_attempts/results/simple_fit_ridge.rda"))

simple_fit_ridge_submission <- bind_cols(test_regression_cleaned,
                                         predict(simple_fit_ridge, test_regression_cleaned)) |>
  select(id, .pred) |>
  rename(predicted = .pred)

simple_fit_ridge_submission <- simple_fit_ridge_submission |>
  mutate(
    predicted = VGAM::yeo.johnson(predicted, lambda = -0.172926, inverse = TRUE) # see recipes script for lambda work
  )

write_csv(simple_fit_ridge_submission, file = here("initial_attempts/submissions/simple_fit_ridge_submission.csv"))

# simple knn----
load(here("initial_attempts/results/simple_fit_knn.rda"))

simple_fit_knn_submission <- bind_cols(test_regression_cleaned,
                                       predict(simple_fit_knn, test_regression_cleaned)) |>
  select(id, .pred) |>
  rename(predicted = .pred)

simple_fit_knn_submission <- simple_fit_knn_submission |>
  mutate(
    predicted = VGAM::yeo.johnson(predicted, lambda = -0.172926, inverse = TRUE) # see recipes script for lambda work
  )

write_csv(simple_fit_knn_submission, file = here("initial_attempts/submissions/simple_fit_knn_submission.csv"))

# simple boosted----
load(here("initial_attempts/results/simple_fit_boosted.rda"))

simple_fit_boosted_submission <- bind_cols(test_regression_cleaned,
                                       predict(simple_fit_boosted, test_regression_cleaned)) |>
  select(id, .pred) |>
  rename(predicted = .pred)

simple_fit_boosted_submission <- simple_fit_boosted_submission |>
  mutate(
    predicted = VGAM::yeo.johnson(predicted, lambda = -0.172926, inverse = TRUE) # see recipes script for lambda work
  )

write_csv(simple_fit_boosted_submission, file = here("initial_attempts/submissions/simple_fit_boosted_submission.csv"))

# simple ensemble----
load(here("initial_attempts/results/simple_model_st.rda"))

simple_ensemble_submission <- bind_cols(test_regression_cleaned, 
                                           predict(simple_model_st, test_regression_cleaned)) |> 
  select(id, .pred) |>
  rename(predicted = .pred)

    # note, untransforming is not necessary bc we fit to the testing set in stacking the model

write_csv(simple_ensemble_submission, file = here("initial_attempts/submissions/simple_ensemble_submission.csv"))

