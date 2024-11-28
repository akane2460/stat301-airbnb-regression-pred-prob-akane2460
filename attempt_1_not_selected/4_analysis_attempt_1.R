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

set.seed(9123)

# ensemble both fit----
load(here("attempt_1_not_selected/results/null_model_st.rda"))

attempt_1_submission <- bind_cols(test_regression_cleaned,
                                      predict(null_model_st,
                                              test_regression_cleaned)) |>
  select(id, .pred) |>
  rename(predicted = .pred)

set.seed(9123)
attempt_1_submission <- attempt_1_submission |>
  mutate(
    predicted = 10^predicted
  )

write_csv(attempt_1_submission, file = here("attempt_1_not_selected/submissions/attempt_1_submission.csv"))

