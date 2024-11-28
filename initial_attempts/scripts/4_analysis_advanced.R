# Analysis advanced

## Load Packages ----
library(tidyverse)
library(tidymodels)
library(skimr)
library(janitor)
library(here)
library(knitr)
library(corrplot)

tidymodels_prefer()

# read in the data----
load(here("data/train_regression_cleaned.rda"))
load(here("data/test_regression_cleaned.rda"))

# # advanced log reg----
# load(here("initial_attempts/results/advanced_fit_lm.rda"))
# 
# advanced_lm_submission <- bind_cols(test_regression_cleaned,
#                                        predict(advanced_fit_lm, test_regression_cleaned)) |>
#   select(id, .pred) |>
#   rename(predicted = .pred)
# 
# advanced_lm_submission <- advanced_lm_submission |>
#   mutate(
#     predicted = 10^predicted
#   )
# 
# write_csv(advanced_lm_submission, file = here("initial_attempts/submissions/advanced_lm_submission.csv"))


# # advanced lasso----
# load(here("initial_attempts/results/advanced_fit_lasso.rda"))
# 
# advanced_fit_lasso_submission <- bind_cols(test_regression_cleaned,
#                                          predict(advanced_fit_lasso, test_regression_cleaned)) |>
#   select(id, .pred) |>
#   rename(predicted = .pred)
# 
# advanced_fit_lasso_submission <- advanced_fit_lasso_submission |>
#   mutate(
#     predicted = VGAM::yeo.johnson(predicted, lambda = -0.172926, inverse = TRUE) # see recipes script for lambda work
#   )
# 
# write_csv(advanced_fit_lasso_submission, file = here("initial_attempts/submissions/advanced_fit_lasso_submission.csv"))
# 
# # advanced ridge----
# load(here("initial_attempts/results/advanced_fit_ridge.rda"))
# 
# advanced_fit_ridge_submission <- bind_cols(test_regression_cleaned,
#                                          predict(advanced_fit_ridge, test_regression_cleaned)) |>
#   select(id, .pred) |>
#   rename(predicted = .pred)
# 
# advanced_fit_ridge_submission <- advanced_fit_ridge_submission |>
#   mutate(
#     predicted = VGAM::yeo.johnson(predicted, lambda = -0.172926, inverse = TRUE) # see recipes script for lambda work
#   )
# 
# write_csv(advanced_fit_ridge_submission, file = here("initial_attempts/submissions/advanced_fit_ridge_submission.csv"))
# 
# # advanced knn----
# load(here("initial_attempts/results/advanced_fit_knn.rda"))
# 
# advanced_fit_knn_submission <- bind_cols(test_regression_cleaned,
#                                        predict(advanced_fit_knn, test_regression_cleaned)) |>
#   select(id, .pred) |>
#   rename(predicted = .pred)
# 
# advanced_fit_knn_submission <- advanced_fit_knn_submission |>
#   mutate(
#     predicted = VGAM::yeo.johnson(predicted, lambda = -0.172926, inverse = TRUE) # see recipes script for lambda work
#   )
# 
# write_csv(advanced_fit_knn_submission, file = here("initial_attempts/submissions/advanced_fit_knn_submission.csv"))
# 
# # advanced boosted----
# load(here("initial_attempts/results/advanced_fit_boosted.rda"))
# 
# advanced_fit_boosted_submission <- bind_cols(test_regression_cleaned,
#                                            predict(advanced_fit_boosted, test_regression_cleaned)) |>
#   select(id, .pred) |>
#   rename(predicted = .pred)
# 
# advanced_fit_boosted_submission <- advanced_fit_boosted_submission |>
#   mutate(
#     predicted = VGAM::yeo.johnson(predicted, lambda = -0.172926, inverse = TRUE) # see recipes script for lambda work
#   )
# 
# write_csv(advanced_fit_boosted_submission, file = here("initial_attempts/submissions/advanced_fit_boosted_submission.csv"))
