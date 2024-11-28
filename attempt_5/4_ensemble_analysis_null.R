# Analysis null attempt 5

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

set.seed(2913)

################################################################################
################################################################################
################################################################################
# ensemble both fit----
load(here("attempt_5/results/ensemble_model_null.rda"))

attempt_5_submission <- bind_cols(test_regression_cleaned,
                                  predict(ensemble_model_null,
                                          test_regression_cleaned)) |>
  select(id, .pred) |>
  rename(predicted = .pred)

attempt_5_submission <- attempt_5_submission |>
  mutate(
    predicted = 10^predicted
  )

# save submission within folder
write_csv(attempt_5_submission, file = here("attempt_5/results/attempt_5_submission.csv"))

# save final submission to final submissions folder
write_csv(attempt_5_submission, file = here("final_submissions/attempt_5_submission.csv"))
################################################################################
################################################################################
################################################################################

# analyzing ensemble stack ----
load(here("attempt_5/results/null_data_stack.rda"))
# explore the blended model stack
autoplot(null_data_stack) +
  theme_minimal()

# visualize how optimal penalty with member coefficients
attempt_5_weights_plot <- autoplot(null_data_stack, type = "weights") +
  theme_minimal()

ggsave(filename = here("plots/attempt_5_weights_plot.png"), attempt_5_weights_plot)

# collect parameters for each component
null_data_stack |> 
  collect_parameters("boosted_tuned_null") |> 
  filter(is.na(coef) == FALSE) |> 
  filter(coef != 0) |> 
  knitr::kable()

null_data_stack |> 
  collect_parameters("knn_tuned_null") |> 
  filter(coef != 0) |>
  knitr::kable()

