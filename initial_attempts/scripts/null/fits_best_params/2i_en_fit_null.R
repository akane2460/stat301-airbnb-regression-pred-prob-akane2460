# Define and fit mars

# random processes present

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)
library(parallel)

# handle common conflicts
tidymodels_prefer()

# parallel processing
num_cores <- parallel::detectCores(logical = TRUE)

registerDoMC(cores = num_cores)

# read in the data----
load(here("data/train_regression_cleaned.rda"))

# load the recipes----
load(here("recipes/null_recipe.rda"))

# load tuned result----
load(here("initial_attempts/results/null_en_tuned.rda"))

# set seed
set.seed(0922221)

# tuned analysis----
# best results
best_results_en <- select_best(null_en_tuned, metric = "mae")
  # penalty = 1.03, mixture = .0236

# model specifications ----
en_spec <- 
  linear_reg(penalty = 1.03, mixture = .0236) |> 
  set_engine("glmnet") |> 
  set_mode("regression") 

# define workflows ----
# null wflow
null_en_wflow <-
  workflow() |> 
  add_model(en_spec) |> 
  add_recipe(null_recipe)

# fitting result---
null_fit_en <- fit(null_en_wflow, train_regression_cleaned)

# write out results----
save(
  null_fit_en,
  file = here("initial_attempts/results/null_fit_en.rda")
)