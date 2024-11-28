# Ensemble model contributors

# Load package(s) ----
library(tidymodels)
library(tidyverse)
library(here)
library(stacks)
library(cli)
library(parallel)
library(doMC)

# handle common conflicts
tidymodels_prefer()

# parallel processing
num_cores <- parallel::detectCores(logical = TRUE)

registerDoMC(cores = num_cores)

# read in the data----
load(here("data/train_regression_cleaned.rda"))

# load the recipes----
load(here("recipes/null_recipe.rda"))

# load folds----
load(here("data/train_folds.rda"))

# set seed
set.seed(265421)

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

# Tuning/fitting
null_lm_res <-
  null_lm_wflow |>
  fit_resamples(
    resamples = train_folds,
    metrics = metric,
    control = ctrl_res
  )

# Write out results & workflow
save(null_lm_res, file = here("initial_attempts/results/null_lm_res.rda"))


# knn result model----
kknn_spec <-
  nearest_neighbor(neighbors = 7, dist_power = 1.12) |>
  set_engine("kknn") |>
  set_mode("regression")

# define workflows
null_knn_wflow <-
  workflow() |>
  add_model(kknn_spec) |>
  add_recipe(null_recipe)

# Tuning/fitting
null_knn_res <-
  null_knn_wflow |>
  fit_resamples(
    resamples = train_folds,
    metrics = metric,
    control = ctrl_res
  )

# Write out results & workflow
save(null_knn_res, file = here("initial_attempts/results/null_knn_res.rda"))

# boosted result model----
boosted_spec <-
  boost_tree(mtry = 13, min_n = 40, learn_rate = 1.02) |>
  set_engine("xgboost") |>
  set_mode("regression")

# define workflows
null_boosted_wflow <-
  workflow() |>
  add_model(boosted_spec) |>
  add_recipe(null_recipe)

# Tuning/fitting
metric <- metric_set(mae)
ctrl_res <- control_stack_resamples()

null_boosted_res <-
  null_boosted_wflow |>
  fit_resamples(
    resamples = train_folds,
    metrics = metric,
    control = ctrl_res
  )

# Write out results & workflow
save(null_boosted_res, file = here("initial_attempts/results/null_boosted_res.rda"))

# nn result model----
nn_spec <-
  mlp(hidden_units = 9, penalty = .00909) |>
  set_mode("regression") |>
  set_engine("nnet")

# define workflows
null_nn_wflow <-
  workflow() |>
  add_model(nn_spec) |>
  add_recipe(null_recipe)

# Tuning/fitting
metric <- metric_set(mae)
ctrl_res <- control_stack_resamples()

null_nn_res <-
  null_nn_wflow |>
  fit_resamples(
    resamples = train_folds,
    metrics = metric,
    control = ctrl_res
  )

# Write out results & workflow
save(null_nn_res, file = here("initial_attempts/results/null_nn_res.rda"))

# svm result model----
svm_spec <-
  svm_rbf(
    cost = .00554, 
    rbf_sigma = 0.0274
  ) |>
  set_mode("regression") |>
  set_engine("kernlab")

# workflow
svm_wflow <-
  workflow() |>
  add_model(svm_spec) |>
  add_recipe(null_recipe)

# Tuning/fitting
metric <- metric_set(mae)
ctrl_res <- control_stack_resamples()

null_svm_res <-
  svm_wflow |>
  fit_resamples(
    resamples = train_folds,
    metrics = metric,
    control = ctrl_res
  )

# Write out results & workflow
save(null_svm_res, file = here("initial_attempts/results/null_svm_res.rda"))

# mars result model----
mars_spec <- mars(
  num_terms = 5,
  prod_degree = 2
) |>
  set_engine("earth") |>
  set_mode("regression")

# define workflows
mars_wflow <-
  workflow() |>
  add_model(mars_spec) |>
  add_recipe(null_recipe)

# Tuning/fitting
metric <- metric_set(mae)
ctrl_res <- control_stack_resamples()

null_mars_res <-
  mars_wflow |>
  fit_resamples(
    resamples = train_folds,
    metrics = metric,
    control = ctrl_res
  )

# Write out results & workflow
save(null_mars_res, file = here("initial_attempts/results/null_mars_res.rda"))

# other considered models----
# no longer exploring
# # # lasso result model
# # model specifications
# lasso_spec <-
#   linear_reg(penalty = 0, mixture = 1) |>
#   set_engine("glmnet", family = "binomial") |>
#   set_mode("regression")
# 
# # null wflow
# null_lasso_wflow <-
#   workflow() |>
#   add_model(lasso_spec) |>
#   add_recipe(null_recipe)
# 
# # Tuning/fitting
# 
# null_lasso_res <-
#   null_lasso_wflow |>
#   fit_resamples(
#     resamples = train_folds,
#     metrics = metric,
#     control = ctrl_res
#   )
# 
# # Write out results & workflow
# save(null_lasso_res, file = here("initial_attempts/results/null_lasso_res.rda"))
# 
# ridge result model
# model specifications
# ridge_spec <-
#   logistic_reg(penalty = 1, mixture = 0) |>
#   set_engine("glmnet", family = "binomial") |>
#   set_mode("regression")
# 
# # null wflow
# null_ridge_wflow <-
#   workflow() |>
#   add_model(ridge_spec) |>
#   add_recipe(null_recipe)
# 
# # Tuning/fitting
# metric <- metric_set(mae)
# ctrl_res <- control_stack_resamples()
# 
# null_ridge_res <-
#   null_ridge_wflow |>
#   fit_resamples(
#     resamples = train_folds,
#     metrics = metric,
#     control = ctrl_res
#   )
# 
# # Write out results & workflow
# save(null_ridge_res, file = here("initial_attempts/results/null_ridge_res.rda"))

# rf result model
# rf_spec <-
#   rand_forest(trees = 2000, min_n = 2, mtry = 14) |>
#   set_engine("ranger") |>
#   set_mode("regression")
# 
# # define workflows
# rf_wflow <-
#   workflow() |>
#   add_model(rf_spec) |>
#   add_recipe(null_recipe)
# 
# # Tuning/fitting
# metric <- metric_set(mae)
# ctrl_res <- control_stack_resamples()
# 
# null_rf_res <-
#   rf_wflow |>
#   fit_resamples(
#     resamples = train_folds,
#     metrics = metric,
#     control = ctrl_res
#   )
# 
# # Write out results & workflow
# save(null_rf_res, file = here("initial_attempts/results/null_rf_res.rda"))