# Variable Selection ----
# Variable selection using lasso

# load package(s) ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)

# handle common conflicts
tidymodels_prefer()

# parallel processing ----
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores - 1)

# create resamples/folds ----
load(here("data/train_regression_cleaned.rda"))

set.seed(0124)
lasso_folds <- 
  train_regression_cleaned |> 
  vfold_cv(v = 5, repeats = 3)


# basic recipe ----
simple_recipe <- recipe(
  price_log10 ~ ., data = train_regression_cleaned) |> 
  step_rm(id, host_since, first_review, last_review) |> 
  step_impute_knn(beds, impute_with = imp_vars(accommodates)) |> 
  step_impute_median(all_numeric_predictors()) |>
  step_impute_mode(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_novel(all_nominal_predictors()) |> 
  step_YeoJohnson(all_predictors()) |>
  step_other(all_nominal_predictors(),
             threshold = .05,
             other = "other")

# checking the recipe 
# simple_recipe |>
#   prep() |>
#   bake(new_data = NULL)

# model specifications ----
lasso_spec <-
  linear_reg(
    mixture = 1,
    penalty = tune()
  ) |> 
  set_mode("regression") |> 
  set_engine("glmnet")

# define workflows ----
lasso_wflow <-
  workflow() |> 
  add_model(lasso_spec) |> 
  add_recipe(simple_recipe)

# hyperparameter tuning values ----
hardhat::extract_parameter_set_dials(lasso_spec)

lasso_params <- hardhat::extract_parameter_set_dials(lasso_spec) |> 
  update(
    penalty = penalty(c(-3, 0))
  )

# build tuning grid
lasso_grid <- grid_regular(lasso_params, levels = 5)

# fit workflow/model ----
lasso_tuned <- 
  lasso_wflow |> 
  tune_grid(
    resamples = lasso_folds, 
    grid = lasso_grid,
    metrics = metric_set(mae),
    control = control_grid(save_workflow = TRUE)
  )

# extract best model (optimal tuning parameters)
optimal_wflow <- 
  extract_workflow(lasso_tuned) |> 
  finalize_workflow(select_best(lasso_tuned, metric = "mae"))

# fit best model/results
var_select_fit_lasso <- fit(optimal_wflow, train_regression_cleaned)

# look at results
var_select_lasso <- var_select_fit_lasso |>  tidy()

var_keep_lasso <- var_select_lasso |> filter(estimate != 0) |> select(term)

var_keep_lasso <- as.list(var_keep_lasso)

var_keep_lasso <- unlist(var_keep_lasso)


var_remove_lasso <- var_select_lasso |> filter(estimate == 0) |> select(term)

var_remove_lasso <- names(test_regression_cleaned |> 
                                 select(-accommodates,
                                        -number_of_reviews_ltm,
                                        -number_of_reviews_l30d,
                                        -review_scores_location,
                                        -calculated_host_listings_count_private_rooms,
                                        -reviews_per_month,
                                        -host_neighbourhood,
                                        -host_verifications,
                                        -neighbourhood_cleansed,
                                        -property_type,
                                        -room_type,
                                        -bathrooms_text,
                                        -id,
                                        -host_since,
                                        -first_review,
                                        -last_review))

# write out variable selection results ----
save(
  var_select_lasso, 
  file = here("results/var_select_lasso.rda")
)

save(
  var_remove_lasso, 
  file = here("results/var_remove_lasso.rda")
)

# vars to keep
# accommodates, 
# number_of_reviews_ltm, 
# number_of_reviews_l30d, 
# review_scores_location,
# calculated_host_listings_count_private_rooms,
# reviews_per_month,
# host_neighourbou
# host_neighbourhood_Clearwater.Beach,
# host_verifications_X..email....phone..,
# neighbourhood_cleansed_Loop,
# neighbourhood_cleansed_Near.North.Side,
# property_type_Entire.home,
# property_type_Private.room.in.home,
# property_type_Private.room.in.rental.unit,
# property_type_Room.in.hotel,
# room_type_Private.room,
# bathrooms_text_X1.shared.bath,
# bathrooms_text_X3.baths,
# bathrooms_text_X3.5.baths


