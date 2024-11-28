# Recipes

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

# load var selected
load(here("results/var_select_lasso.rda"))

# Yeo Johnson lambda
yj_recipe <- recipe(price_log10 ~., data = train_regression_cleaned) |>
  step_YeoJohnson(price_log10, skip = TRUE)

yj_fit <- yj_recipe |>
  prep() |>
  bake(new_data = NULL) |>
  rename(price_log10_yj = price_log10)

lambda <- yj_recipe |>
  prep() |>
  tidy(1) |>
  knitr::kable()
  # optimal lambda value is -0.1346749

# recipes----
# null
null_recipe <- recipe(
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
             other = "other") |> 
  step_nzv(all_predictors()) |>
  step_normalize(all_predictors())

# prep(null_recipe) |>
#   bake(new_data = NULL)

save(null_recipe, file = here("recipes/null_recipe.rda"))

# simple recipe
simple_recipe <- recipe(
  price_log10 ~ ., data = train_regression_cleaned) |> 
  step_rm(id, host_since, first_review, last_review) |> 
  step_impute_knn(beds, impute_with = imp_vars(accommodates)) |> 
  step_impute_median(all_numeric_predictors()) |>
  step_impute_mode(all_nominal_predictors()) |>
  step_novel(all_nominal_predictors()) |> 
  step_interact(terms = ~ review_scores_rating : starts_with("review_scores")) |> 
  step_interact(terms = ~ beds : accommodates) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_YeoJohnson(all_predictors()) |> 
  step_other(all_nominal_predictors(),
             threshold = .05,
             other = "other") |> 
  step_nzv(all_predictors()) |>
  step_normalize(all_predictors())

# prep(simple_recipe) |>
#   bake(new_data = NULL)

save(simple_recipe, file = here("recipes/simple_recipe.rda"))


# advanced
advanced_recipe <- recipe(
  price_log10 ~ ., data = train_regression_cleaned) |> 
  step_rm(id, host_since, first_review, last_review) |> 
  step_impute_knn(beds, impute_with = imp_vars(accommodates)) |> 
  step_impute_knn(bathrooms_text, impute_with = imp_vars(beds)) |> 
  step_impute_median(all_numeric_predictors()) |>
  step_impute_mode(all_nominal_predictors()) |>
  step_novel(all_nominal_predictors()) |> 
  step_interact(terms = ~ review_scores_rating : starts_with("review_scores")) |> 
  step_interact(terms = ~ beds : accommodates) |> 
  step_interact(terms = ~ availability_90:availability_365) |> 
  step_interact(terms = ~ property_type:room_type) |> 
  step_interact(terms = ~ beds:bathrooms_text) |> 
  step_interact(terms = ~ accommodates:property_type) |> 
  step_interact(terms = ~ calculated_host_listings_count_private_rooms:property_type) |> 
  step_interact(terms = ~ review_scores_location:neighbourhood_cleansed) |> 
  step_interact(terms = ~ longitude:latitude) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_YeoJohnson(all_predictors()) |> 
  step_other(all_nominal_predictors(),
             threshold = .05,
             other = "other")  |> 
  step_nzv(all_predictors()) |>
  step_normalize(all_predictors())

# prep(advanced_recipe) |>
#   bake(new_data = NULL)

save(advanced_recipe, file = here("recipes/advanced_recipe.rda"))

