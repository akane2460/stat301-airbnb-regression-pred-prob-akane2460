# Analysis null

## Load Packages ----
library(tidyverse)
library(tidymodels)
library(skimr)
library(janitor)
library(here)
library(knitr)
library(corrplot)
library(doMC)
library(parallel)

# handle common conflicts
tidymodels_prefer()

# parallel processing
num_cores <- parallel::detectCores(logical = TRUE)

registerDoMC(cores = num_cores)

# read in the data----
load(here("data/train_regression_cleaned.rda"))
load(here("data/test_regression_cleaned.rda"))

# # ensemble both fit----
# load(here("initial_attempts/results/null_model_st.rda"))
# 
# null_ensemble_submission <- bind_cols(test_regression_cleaned,
#             predict(null_model_st,
#                     test_regression_cleaned)) |>
#   select(id, .pred) |>
#   rename(predicted = .pred)
# 
# null_ensemble_submission <- null_ensemble_submission |>
#     mutate(
#       predicted = 10^predicted
#     )
# 
# write_csv(null_ensemble_submission, file = here("initial_attempts/submissions/null_ensemble_submission.csv"))

# other analyses----
# no longer exploring
# # null lm
# load(here("initial_attempts/results/null_fit_lm.rda"))
# 
# null_lm_submission <- bind_cols(test_regression_cleaned,
#                                        predict(null_fit_lm, test_regression_cleaned)) |>
#   select(id, .pred) |>
#   rename(predicted = .pred)
# 
# null_lm_submission <- null_lm_submission |>
#   mutate(
#     predicted = 10^predicted
#   )
# 
# write_csv(null_lm_submission, file = here("initial_attempts/submissions/null_lm_submission.csv"))
# 
# 
# # null en
# load(here("initial_attempts/results/null_fit_en.rda"))
# 
# null_en_submission <- bind_cols(test_regression_cleaned,
#                                       predict(null_fit_en,
#                                               test_regression_cleaned)) |>
#   select(id, .pred) |>
#   rename(predicted = .pred)
# 
# null_en_submission <- null_en_submission |>
#   mutate(
#     predicted = 10^predicted
#   )
# 
# write_csv(null_en_submission, file = here("initial_attempts/submissions/null_en_submission.csv"))
# 
# 
# null lasso
load(here("initial_attempts/results/null_fit_lasso.rda"))

null_fit_lasso_submission <- bind_cols(test_regression_cleaned,
                                         predict(null_fit_lasso, test_regression_cleaned)) |>
  select(id, .pred) |>
  rename(predicted = .pred)

null_fit_lasso_submission <- null_fit_lasso_submission |>
  mutate(
    predicted = VGAM::yeo.johnson(predicted, lambda = -0.172926, inverse = TRUE) # see recipes script for lambda work
  )

write_csv(null_fit_lasso_submission, file = here("initial_attempts/submissions/null_fit_lasso_submission.csv"))
#
# # null ridge
# load(here("initial_attempts/results/null_fit_ridge.rda"))
#
# null_fit_ridge_submission <- bind_cols(test_regression_cleaned,
#                                          predict(null_fit_ridge, test_regression_cleaned)) |>
#   select(id, .pred) |>
#   rename(predicted = .pred)
#
# null_fit_ridge_submission <- null_fit_ridge_submission |>
#   mutate(
#     predicted = VGAM::yeo.johnson(predicted, lambda = -0.172926, inverse = TRUE) # see recipes script for lambda work
#   )
#
# write_csv(null_fit_ridge_submission, file = here("initial_attempts/submissions/null_fit_ridge_submission.csv"))

# null knn
# load(here("initial_attempts/results/null_fit_knn.rda"))
#
# null_fit_knn_submission <- bind_cols(test_regression_cleaned,
#                                        predict(null_fit_knn, test_regression_cleaned)) |>
#   select(id, .pred) |>
#   rename(predicted = .pred)
#
# null_fit_knn_submission <- null_fit_knn_submission |>
#   mutate(
#     predicted = VGAM::yeo.johnson(predicted, lambda = -0.172926, inverse = TRUE) # see recipes script for lambda work
#   )

# write_csv(null_fit_knn_submission, file = here("initial_attempts/submissions/null_fit_knn_submission.csv"))

# # null boosted
# # load(here("initial_attempts/results/null_fit_boosted.rda"))
# # 
# # null_fit_boosted_submission <- bind_cols(test_regression_cleaned,
# #                                            predict(null_fit_boosted, test_regression_cleaned)) |>
# #   select(id, .pred) |>
# #   rename(predicted = .pred)
# # 
# # null_fit_boosted_submission <- null_fit_boosted_submission |>
# #   mutate(
# #     predicted = VGAM::yeo.johnson(predicted, lambda = -0.172926, inverse = TRUE) # see recipes script for lambda work
# #   )
# # 
# # write_csv(null_fit_boosted_submission, file = here("initial_attempts/submissions/null_fit_boosted_submission.csv"))
# 
# # null nn
# # load(here("initial_attempts/results/null_fit_nn.rda"))
# # 
# # null_nn_submission <- bind_cols(test_regression_cleaned,
# #                                          predict(null_fit_nn, test_regression_cleaned)) |>
# #   select(id, .pred) |>
# #   rename(predicted = .pred)
# # 
# # null_nn_submission<- null_nn_submission |>
# #   mutate(
# #     predicted = VGAM::yeo.johnson(predicted, lambda = -0.172926, inverse = TRUE) # see recipes script for lambda work
# #   )
# # 
# # write_csv(null_nn_submission, file = here("initial_attempts/submissions/null_nn_submission.csv"))
