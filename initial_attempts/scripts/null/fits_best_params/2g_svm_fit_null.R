# Tune SVM RBF model

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)
library(stacks)
library(cli)


# Handle common conflicts
tidymodels_prefer()

# parallel processing ----
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores)

# read in the data----
load(here("data/train_regression_cleaned.rda"))

# load the recipes----
load(here("recipes/null_recipe.rda"))

# load tuned result----
load(here("initial_attempts/results/null_svm_tuned.rda"))

# set seed
set.seed(0997391)

# tuned analysis----
# best results
best_results_svm <- select_best(null_svm_tuned, metric = "mae")
# cost = .00554, rbf_signma = 0.0274 

# # model specification ----
# svm_spec <-
#   svm_rbf(
#     cost = 0.294, rbf_signma = 0.00000809
#   ) |>
#   set_mode("regression") |>
#   set_engine("kernlab")
# 
# # workflow ----
# svm_wflow <-
#   workflow() |>
#   add_model(svm_spec) |>
#   add_recipe(null_recipe)
# 
# # Tuning/fitting ----
# null_fit_svm <- fit(svm_wflow, train_regression_cleaned)
# 
# # Write out results & workflow
# save(null_fit_svm, file = here("initial_attempts/results/null_fit_svm.rda"))