# Ensemble model contributors

# Load package(s) ----
library(tidymodels)
library(tidyverse)
library(here)
library(stacks)
library(cli)
library(doMC)

# Handle common conflicts
tidymodels_prefer()

# parallel processing
num_cores <- parallel::detectCores(logical = TRUE)

registerDoMC(cores = num_cores)

# read in the data----
load(here("data/train_regression_cleaned.rda"))

# load the recipes----
load(here("recipes/simple_recipe.rda"))

# load folds----
load(here("data/train_folds.rda"))

# set seed
set.seed(209174)

# log reg result model----
# model specifications
lm_spec <-
  linear_reg() |>
  set_engine("lm") |>
  set_mode("regression")

# define workflows ----
simple_lm_wflow <-
  workflow() |>
  add_model(lm_spec) |>
  add_recipe(simple_recipe)

# Tuning/fitting
metric <- metric_set(mae)
ctrl_res <- control_stack_resamples()

simple_lm_res <-
  simple_lm_wflow|>
  fit_resamples(
    resamples = train_folds,
    metrics = metric,
    control = ctrl_res
  )

# Write out results & workflow
save(simple_lm_res, file = here("initial_attempts/results/simple_lm_res.rda"))


# lasso result model----
# model specifications
lasso_spec <-
  linear_reg(penalty = 0, mixture = 1) |>
  set_engine("glmnet") |>
  set_mode("regression")

# define workflows ----
# simple wflow
simple_lasso_wflow <-
  workflow() |>
  add_model(lasso_spec) |>
  add_recipe(simple_recipe)

# Tuning/fitting
metric <- metric_set(mae)
ctrl_res <- control_stack_resamples()

simple_lasso_res <-
  simple_lasso_wflow |>
  fit_resamples(
    resamples = train_folds,
    metrics = metric,
    control = ctrl_res
  )

# Write out results & workflow
save(simple_lasso_res, file = here("initial_attempts/results/simple_lasso_res.rda"))

# ridge result model----
# model specifications
ridge_spec <-
  linear_reg(penalty = 1, mixture = 0) |>
  set_engine("glmnet") |>
  set_mode("regression")

# define workflows ----
# simple wflow
simple_ridge_wflow <-
  workflow() |>
  add_model(ridge_spec) |>
  add_recipe(simple_recipe)

# Tuning/fitting
metric <- metric_set(mae)
ctrl_res <- control_stack_resamples()

simple_ridge_res <-
  simple_ridge_wflow |>
  fit_resamples(
    resamples = train_folds,
    metrics = metric,
    control = ctrl_res
  )

# Write out results & workflow
save(simple_ridge_res, file = here("initial_attempts/results/simple_ridge_res.rda"))

# knn result model----
kknn_spec <-
  nearest_neighbor(neighbors = 2, dist_power = 2.91) |>
  set_engine("kknn") |>
  set_mode("regression")

# define workflows 
simple_knn_wflow <-
  workflow() |>
  add_model(kknn_spec) |>
  add_recipe(simple_recipe)

# Tuning/fitting
metric <- metric_set(mae)
ctrl_res <- control_stack_resamples()

simple_knn_res <-
  simple_knn_wflow |>
  fit_resamples(
    resamples = train_folds,
    metrics = metric,
    control = ctrl_res
  )

# Write out results & workflow
save(simple_knn_res, file = here("initial_attempts/results/simple_knn_res.rda"))

# boosted result model----
boosted_spec <-
  boost_tree(mtry = 11, min_n = 2, learn_rate = 1.05) |>
  set_engine("xgboost") |>
  set_mode("regression")

# define workflows 
simple_boosted_wflow <-
  workflow() |>
  add_model(boosted_spec) |>
  add_recipe(simple_recipe)

# Tuning/fitting
metric <- metric_set(mae)
ctrl_res <- control_stack_resamples()

simple_boosted_res <-
  simple_boosted_wflow |>
  fit_resamples(
    resamples = train_folds,
    metrics = metric,
    control = ctrl_res
  )

# Write out results & workflow
save(simple_boosted_res, file = here("initial_attempts/results/simple_boosted_res.rda"))

# nn result model----
nn_spec <-
  mlp(hidden_units = 7, penalty = 1) |>
  set_mode("regression") |>
  set_engine("nnet")

# define workflows 
simple_nn_wflow <-
  workflow() |>
  add_model(nn_spec) |>
  add_recipe(simple_recipe)

# Tuning/fitting
metric <- metric_set(mae)
ctrl_res <- control_stack_resamples()

simple_nn_res <-
  simple_nn_wflow |>
  fit_resamples(
    resamples = train_folds,
    metrics = metric,
    control = ctrl_res
  )

# Write out results & workflow
save(simple_nn_res, file = here("initial_attempts/results/simple_nn_res.rda"))

# svm result model----
svm_spec <-
  svm_rbf(
    cost = 32,
    rbf_sigma = .00316
  ) |>
  set_mode("regression") |>
  set_engine("kernlab")

# workflow 
svm_wflow <-
  workflow() |>
  add_model(svm_spec) |>
  add_recipe(simple_recipe)

# Tuning/fitting
metric <- metric_set(mae)
ctrl_res <- control_stack_resamples()

simple_svm_res <-
  svm_wflow |>
  fit_resamples(
    resamples = train_folds,
    metrics = metric,
    control = ctrl_res
  )

# Write out results & workflow
save(simple_svm_res, file = here("initial_attempts/results/simple_svm_res.rda"))

# # rf result model----
# rf_spec <-
#   rand_forest(trees = 1, min_n = 11, mtry = 7) |>
#   set_engine("ranger") |>
#   set_mode("regression")
# 
# # define workflows
# rf_wflow <-
#   workflow() |>
#   add_model(rf_spec) |>
#   add_recipe(simple_recipe)
# 
# # Tuning/fitting
# metric <- metric_set(mae)
# ctrl_res <- control_stack_resamples()
# 
# simple_rf_res <-
#   rf_wflow |>
#   fit_resamples(
#     resamples = train_folds,
#     metrics = metric,
#     control = ctrl_res
#   )
# 
# # Write out results & workflow
# save(simple_rf_res, file = here("initial_attempts/results/simple_rf_res.rda"))

# mars result model----
mars_spec <- mars(
  num_terms = 3,
  prod_degree = 1
) |>
  set_mode("regression") |>
  set_engine("earth")

# define workflow ----
simple_mars_wflow <-
  workflow() |>
  add_model(mars_spec) |>
  add_recipe(simple_recipe)

# Tuning/fitting
metric <- metric_set(mae)
ctrl_res <- control_stack_resamples()

simple_mars_res <-
  simple_mars_wflow |>
  fit_resamples(
    resamples = train_folds,
    metrics = metric,
    control = ctrl_res
  )

# Write out results & workflow
save(simple_mars_res, file = here("initial_attempts/results/simple_mars_res.rda"))
