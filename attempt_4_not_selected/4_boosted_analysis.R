# Analysis null attempt 1

## Load Packages ----
library(tidyverse)
library(tidymodels)
library(skimr)
library(janitor)
library(here)
library(knitr)
library(corrplot)

# handle common conflicts
tidymodels_prefer()

# read in the data----
load(here("data/train_regression_cleaned.rda"))
load(here("data/test_regression_cleaned.rda"))

# ensemble both fit----
load(here("attempt_4/results/null_fit_boosted.rda"))

attempt_4_submission <- bind_cols(test_regression_cleaned,
                                  predict(null_fit_boosted,
                                          test_regression_cleaned)) |>
  select(id, .pred) |>
  rename(predicted = .pred)

attempt_4_submission <- attempt_4_submission|>
  mutate(
    predicted = 10^predicted
  )

write_csv(attempt_4_submission, file = here("attempt_4/results/attempt_4_submission.csv"))