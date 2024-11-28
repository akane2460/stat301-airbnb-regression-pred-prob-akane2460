# Ensemble model contributors

# Load package(s) ----
library(tidymodels)
library(tidyverse)
library(here)
library(stacks)
library(cli)

# handle common conflicts
tidymodels_prefer()

# read in the data----
load(here("data/train_regression_cleaned.rda"))

# load the recipes----
load(here("recipes/null_recipe.rda"))

# load folds----
load(here("data/train_folds.rda"))

# set seed
set.seed(0923)

metric <- metric_set(mae)
ctrl_res <- control_stack_resamples()

# lm result model----
# model specifications
lm_spec <-
  linear_reg() |>
  set_engine("glm", maxit = 100000) |>
  set_mode("regression")

# null wflow
null_lm_wflow <-
  workflow() |>
  add_model(lm_spec) |>
  add_recipe(null_recipe)

set.seed(0923)
# Tuning/fitting
null_lm_res <-
  null_lm_wflow |>
  fit_resamples(
    resamples = train_folds,
    metrics = metric,
    control = ctrl_res
  )

# Write out results & workflow
save(null_lm_res, file = here("attempt_1_not_selected/results/null_lm_res.rda"))

# # knn result model----
# set seed
set.seed(0923)

# model specification ----
knn_spec <-
  nearest_neighbor(
    neighbors = tune(),
    dist_power = tune()
  ) |>
  set_mode("regression") |>
  set_engine("kknn")

# set-up tuning grid ----
knn_params <- hardhat::extract_parameter_set_dials(knn_spec) |>
  update(neighbors = neighbors(range = c(1, 20)),
         dist_power = dist_power(range = c(0, 2)
         ))

# set seed
set.seed(0923)

# define grid
knn_grid <- grid_latin_hypercube(knn_params, size = 10)

# workflow ----
knn_wflow <-
  workflow() |>
  add_model(knn_spec) |>
  add_recipe(null_recipe)

# Tuning/fitting ----

metric <- metric_set(mae)
ctrl_res <- control_stack_resamples()

# set seed
set.seed(0923)

knn_res <-
  knn_wflow |>
  tune_grid(
    resamples = train_folds,
    grid = knn_grid,
    metrics = metric,
    control = ctrl_res
  )

# Write out results & workflow
save(knn_res, file = here("results/knn_res.rda"))

# Write out results & workflow
save(null_knn_res, file = here("attempt_1_not_selected/results/null_knn_res.rda"))


# # Write out results & workflow
# save(null_knn_res, file = here("attempt_1_not_selected/results/null_knn_res.rda"))
# 
# # boosted result model----
# boosted_spec <-
#   boost_tree(mtry = 1, min_n = 2, learn_rate = 1.02) |>
#   set_engine("xgboost") |>
#   set_mode("regression")
# 
# # define workflows
# null_boosted_wflow <-
#   workflow() |>
#   add_model(boosted_spec) |>
#   add_recipe(null_recipe)
# 
# # Tuning/fitting
# metric <- metric_set(mae)
# ctrl_res <- control_stack_resamples()
# 
# set.seed(265421)
# null_boosted_res <-
#   null_boosted_wflow |>
#   fit_resamples(
#     resamples = train_folds,
#     metrics = metric,
#     control = ctrl_res
#   )
# 
# # Write out results & workflow
# save(null_boosted_res, file = here("attempt_1_not_selected/results/null_boosted_res.rda"))
# 
# # nn result model----
# nn_spec <-
#   mlp(hidden_units = 1, penalty = 0.00000681) |>
#   set_mode("regression") |>
#   set_engine("nnet")
# 
# # define workflows
# null_nn_wflow <-
#   workflow() |>
#   add_model(nn_spec) |>
#   add_recipe(null_recipe)
# 
# # Tuning/fitting
# metric <- metric_set(mae)
# ctrl_res <- control_stack_resamples()
# 
# set.seed(265421)
# null_nn_res <-
#   null_nn_wflow |>
#   fit_resamples(
#     resamples = train_folds,
#     metrics = metric,
#     control = ctrl_res
#   )
# 
# # Write out results & workflow
# save(null_nn_res, file = here("attempt_1_not_selected/results/null_nn_res.rda"))
# 
# # svm result model----
# svm_spec <-
#   svm_rbf(
#     cost = 0.294,
#     rbf_sigma = 0.00000809
#   ) |>
#   set_mode("regression") |>
#   set_engine("kernlab")
# 
# # workflow
# svm_wflow <-
#   workflow() |>
#   add_model(svm_spec) |>
#   add_recipe(null_recipe)
# 
# # Tuning/fitting
# metric <- metric_set(mae)
# ctrl_res <- control_stack_resamples()
# 
# set.seed(265421)
# null_svm_res <-
#   svm_wflow |>
#   fit_resamples(
#     resamples = train_folds,
#     metrics = metric,
#     control = ctrl_res
#   )
# 
# # Write out results & workflow
# save(null_svm_res, file = here("attempt_1_not_selected/results/null_svm_res.rda"))
# 
# # mars result model----
# mars_spec <- mars(
#   num_terms = 5,
#   prod_degree = 2
# ) |>
#   set_engine("earth") |>
#   set_mode("regression")
# 
# # define workflows
# mars_wflow <-
#   workflow() |>
#   add_model(mars_spec) |>
#   add_recipe(null_recipe)
# 
# # Tuning/fitting
# metric <- metric_set(mae)
# ctrl_res <- control_stack_resamples()
# 
# set.seed(265421)
# null_mars_res <-
#   mars_wflow |>
#   fit_resamples(
#     resamples = train_folds,
#     metrics = metric,
#     control = ctrl_res
#   )
# 
# # Write out results & workflow
# save(null_mars_res, file = here("attempt_1_not_selected/results/null_mars_res.rda"))
