# Data Familiarization

## Load Packages ----
library(tidyverse)
library(tidymodels)
library(skimr)
library(janitor)
library(here)
library(knitr)
library(corrplot)
library(caret)

tidymodels_prefer()

# read in the data----
train_regression <- read_csv(here("data/train_regression.csv"),
                             col_types = cols(id = col_character()))
test_regression <- read_csv(here("data/test_regression.csv"),
                            col_types = cols(id = col_character())
                            )

# general data skim
# skimr::skim_without_charts(train_regression)

# missing data in host_location, host_response_time, host_response_rate, 
# host_neighborhood, host_acceptance_rate, bathrooms_text, first_review
# last_review, beds, review_scores_rating, review_scores_accuracy, 
# review_scores_cleanliness, review_scores_checkin, review_scores_communication
# review_scores_location, review_scores_value, reviews_per_month

# cleaning training----
# price
train_regression$price <- gsub("[^0-9.]", "", train_regression$price)

train_regression$price <- as.numeric(train_regression$price)

# pricelog10
train_regression <- train_regression |> 
  mutate(price_log10 = log10(price)) |> 
  select(-price)

# host_is_superhost
train_regression$host_is_superhost <- as.factor(train_regression$host_is_superhost)

# host_has_profile_pic
train_regression$host_has_profile_pic <- as.factor(train_regression$host_has_profile_pic)

# host_identity_verified
train_regression$host_identity_verified <- as.factor(train_regression$host_identity_verified)

# has_availability
train_regression$has_availability <- as.factor(train_regression$host_is_superhost)

# instant_bookable
train_regression$instant_bookable <- as.factor(train_regression$instant_bookable)

# host_location
train_regression$host_location <- 
  as.factor(train_regression$host_location)

# host_response_time  
train_regression$host_response_time <- 
  as.factor(train_regression$host_response_time)

# host_neighbourhood 
train_regression$host_neighbourhood <- 
  as.factor(train_regression$host_neighbourhood)

# host_verifications
train_regression$host_verifications <- 
  as.factor(train_regression$host_verifications)

# neighbourhood_cleansed 
train_regression$neighbourhood_cleansed <- 
  as.factor(train_regression$neighbourhood_cleansed)

# property_type 
train_regression$property_type <- 
  as.factor(train_regression$property_type)

# room_type 
train_regression$room_type <- 
  as.factor(train_regression$room_type)

# host_response_rate
train_regression$host_response_rate <- 
  gsub("[^0-9.]", "", train_regression$host_response_rate)

train_regression$host_response_rate <- 
  as.numeric(train_regression$host_response_rate)

# bathrooms_text
train_regression$bathrooms_text <- 
  gsub("[^0-9.]", "", train_regression$bathrooms_text)

train_regression$bathrooms_text <- 
  as.numeric(train_regression$bathrooms_text)

# host_acceptance_rate  
train_regression$host_acceptance_rate <- 
  gsub("[^0-9.]", "", train_regression$host_acceptance_rate)

train_regression$host_acceptance_rate <- 
  as.numeric(train_regression$host_acceptance_rate)

# host_since year
train_regression$host_since <- year(train_regression$host_since)

# first_review year
train_regression$first_review <- year(train_regression$first_review)

# last_review year
train_regression$last_review <- year(train_regression$last_review)

# saving cleaned data
train_regression_cleaned <- train_regression

train_regression_cleaned |> skimr::skim_without_charts()

save(train_regression_cleaned, file = here("data/train_regression_cleaned.rda"))

# cleaning testing----
# host_is_superhost
test_regression$host_is_superhost <- as.factor(test_regression$host_is_superhost)

# host_has_profile_pic
test_regression$host_has_profile_pic <- as.factor(test_regression$host_has_profile_pic)

# host_identity_verified
test_regression$host_identity_verified <- as.factor(test_regression$host_identity_verified)

# has_availability
test_regression$has_availability <- as.factor(test_regression$host_is_superhost)

# instant_bookable
test_regression$instant_bookable <- as.factor(test_regression$instant_bookable)

# host_location
test_regression$host_location <- 
  as.factor(test_regression$host_location)

# host_response_time  
test_regression$host_response_time <- 
  as.factor(test_regression$host_response_time)

# host_neighbourhood 
test_regression$host_neighbourhood <- 
  as.factor(test_regression$host_neighbourhood)

# host_verifications
test_regression$host_verifications <- 
  as.factor(test_regression$host_verifications)

# neighbourhood_cleansed 
test_regression$neighbourhood_cleansed <- 
  as.factor(test_regression$neighbourhood_cleansed)

# property_type 
test_regression$property_type <- 
  as.factor(test_regression$property_type)

# room_type 
test_regression$room_type <- 
  as.factor(test_regression$room_type)

# bathrooms_text 
test_regression$bathrooms_text <- 
  as.factor(test_regression$bathrooms_text)

# host_response_rate
test_regression$host_response_rate <- 
  gsub("[^0-9.]", "", test_regression$host_response_rate)

test_regression$host_response_rate <- 
  as.numeric(test_regression$host_response_rate)

# bathrooms_text
test_regression$bathrooms_text <- 
  gsub("[^0-9.]", "", test_regression$bathrooms_text)

test_regression$bathrooms_text <- 
  as.numeric(test_regression$bathrooms_text)

# host_acceptance_rate  
test_regression$host_acceptance_rate <- 
  gsub("[^0-9.]", "", test_regression$host_acceptance_rate)

test_regression$host_acceptance_rate <- 
  as.numeric(test_regression$host_acceptance_rate)

# host_since year
test_regression$host_since <- year(test_regression$host_since)

# first_review year
test_regression$first_review <- year(test_regression$first_review)

# last_review year
test_regression$last_review <- year(test_regression$last_review)

# save cleaned testing data
test_regression_cleaned <- test_regression

test_regression_cleaned |> skimr::skim_without_charts()

save(test_regression_cleaned, file = here("data/test_regression_cleaned.rda"))


# train folds----
set.seed(0923)
train_folds <- train_regression_cleaned |>
  vfold_cv(v = 5, repeats = 3)

save(train_folds, file = here("data/train_folds.rda"))


# eda----

# target variable exploration
log10price_dist_plot <- train_regression |>
  ggplot(aes(x = price_log10)) +
  geom_histogram() +
  labs(
    title = "Airbnb Prices",
    x = "Log 10 Price",
    y = "Count"
  ) +
  theme_minimal()


# searching for potential interactions----

# important features identified in 2_var_select_rf only considered:
# beds + 
# accommodates + 
# property_type +
# room_type +
# longitude + 
# latitude + 
# review_scores_rating + 
# calculated_host_listings_count_private_rooms + 
# review_scores_location + 
# reviews_per_month + 
# bathrooms_text_X1.bath + 
# host_since + 
# neighbourhood_cleansed + 
# availability_90 + 
# availability_365 + 
# review_scores_checkin + 
# review_scores_cleanliness
#   property_type +
#   room_type +
#   bathrooms_text


# interactions found: 
# beds: accomodates
# longitude : latitude
# review_scores_rating : review_scores_location , review_scores_cleanliness, review_scores_checkin 
# review_scores_location : review_scores_cleanliness, review_scores_checkin 
# review_scores_cleanliness: review_scores_checkin
# availability_90 : availability_365
#  property_type:room_type
# beds:bathrooms_text
# accommodates:property_type
# calculated_host_listings_count_private_rooms:property_type
# review_scores_location:neighbourhood_cleansed
  
# weaker interactions
  # longitude:neighbourhood_cleansed
  # latitude:neighbourhood_cleansed



# numeric x numeric
train_numeric <- train_regression_cleaned |> 
  select(beds, accommodates,
    longitude, latitude,
    review_scores_rating,
    calculated_host_listings_count_private_rooms,
    review_scores_location,
    reviews_per_month, host_since,
    review_scores_checkin, review_scores_cleanliness,
     availability_90, availability_365)

train_numeric <- na.omit(train_numeric)

correlation_matrix <- cor(train_numeric) 

corrplot <- corrplot(correlation_matrix)

# potential interactions
# beds: accomodates
# longitude : latitude
# review_scores_rating : review_scores_location , review_scores_cleanliness, review_scores_checkin 
# review_scores_location : review_scores_cleanliness, review_scores_checkin 
# review_scores_cleanliness: review_scores_checkin
# availability_90 : availability_365


# factor x factor----

# interactions:
  #  property_type:room_type

train_regression_cleaned |> 
ggplot(aes(x = property_type, fill = room_type)) +
  geom_bar() 
  # property_type and room_type definitely have a relationship

train_regression_cleaned |> 
  ggplot(aes(x = property_type)) +
  geom_bar() +
  facet_wrap(~ bathrooms_text)
  # not much of a relationship here

train_regression_cleaned |> 
  ggplot(aes(x = bathrooms_text, fill = room_type)) +
  geom_bar() 
   # not a major relationship here

# factor x numeric-----

# interactions: 
  # beds:bathrooms_text
  # accommodates:property_type
  # calculated_host_listings_count_private_rooms:property_type
  # review_scores_location:neighbourhood_cleansed

  # longitude:neighbourhood_cleansed
  # latitude:neighbourhood_cleansed

train_regression_cleaned |> 
  ggplot(aes(x = review_scores_location)) +
  geom_boxplot() +
  facet_wrap(~ neighbourhood_cleansed)
    # DEF an interaction, varies from place to place

train_regression_cleaned |> 
  ggplot(aes(x = latitude)) +
  geom_boxplot() +
  facet_wrap(~ neighbourhood_cleansed)
    # some interaction

train_regression_cleaned |> 
  ggplot(aes(x = longitude)) +
  geom_boxplot() +
  facet_wrap(~ neighbourhood_cleansed)
    # some interaction

train_regression_cleaned |> 
  ggplot(aes(x = calculated_host_listings_count_private_rooms)) +
  geom_boxplot() +
  facet_wrap(~ property_type)
  # def an interaction

train_regression_cleaned |> 
  ggplot(aes(x = beds)) +
  geom_boxplot() +
  facet_wrap(~ property_type)

train_regression_cleaned |> 
  ggplot(aes(x = accommodates)) +
  geom_boxplot() +
  facet_wrap(~ property_type)
    # def an interaction

train_regression_cleaned |> 
  ggplot(aes(x = beds)) +
  geom_boxplot() +
  facet_wrap(~ bathrooms_text)
    # DEF an interaction


