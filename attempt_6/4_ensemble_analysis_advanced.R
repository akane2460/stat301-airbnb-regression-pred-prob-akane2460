# Analysis advanced attempt 5

## Load Packages ----
library(tidyverse)
library(tidymodels)
library(skimr)
library(janitor)
library(here)
library(knitr)
library(corrplot)
library(stacks)

# handle common conflicts
tidymodels_prefer()

# read in the data----
load(here("data/train_regression_cleaned.rda"))
load(here("data/test_regression_cleaned.rda"))

set.seed(8463)

################################################################################
################################################################################
################################################################################
# ensemble both fit----
load(here("attempt_6/results/ensemble_model_advanced.rda"))

attempt_6_submission <- bind_cols(test_regression_cleaned,
                                  predict(ensemble_model_advanced,
                                          test_regression_cleaned)) |>
  select(id, .pred) |>
  rename(predicted = .pred)

attempt_6_submission <- attempt_6_submission |>
  mutate(
    predicted = 10^predicted
  )

# save final submission to results folder
write_csv(attempt_6_submission, file = here("attempt_6/results/attempt_6_submission.csv"))

# save final submission to final submissions folder
write_csv(attempt_6_submission, file = here("final_submissions/attempt_6_submission.csv"))

################################################################################
################################################################################
################################################################################

# analyzing ensemble stack ----
load(here("attempt_6/results/advanced_data_stack.rda"))
# explore the blended model stack
autoplot(advanced_data_stack) +
  theme_minimal()

# visualize how optimal penalty with member coefficients
attempt_6_weights_plot <- autoplot(advanced_data_stack, type = "weights") +
  theme_minimal()

ggsave(filename = here("plots/attempt_6_weights_plot.png"), attempt_6_weights_plot)

# collect parameters for each component
advanced_data_stack |> 
  collect_parameters("boosted_tuned_advanced") |> 
  filter(is.na(coef) == FALSE) |> 
  filter(coef != 0) |> 
  knitr::kable()

advanced_data_stack |> 
  collect_parameters("knn_tuned_advanced") |> 
  # filter(coef != 0) |>
  knitr::kable()

