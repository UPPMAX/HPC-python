#!/bin/env Rscript

# Goal: determine the correlation between
# course satisfaction and confidence outcomes
#
#
# |  confidence
# |
# |
# |
# +-----------------
#             satisfaction
#


#' Create non-correlated and correlated values
create_test_values <- function(n = 50) {
  t <- tibble::tibble(
    average_confidence = runif(n = n, min = 0.0, max = 5.0),
    random_satisfactions = runif(n = n, min = 1.0, max = 10.0)
  )
  t$correlated_satisfactions <- t$average_confidence * 2
  t
}
testthat::expect_true(nrow(create_test_values()) > 1)

#' Determine if the two columns are correlated
are_correlated <- function(t, alpha_value = 0.05) {
  testthat::expect_equal(ncol(t), 2)
  results <- correlation::correlation(t)
  results$p < alpha_value
}

testthat::expect_false(
  are_correlated(
    t = create_test_values() |> dplyr::select(average_confidence, random_satisfactions)
  )
)
testthat::expect_true(
  are_correlated(
    t = create_test_values() |> dplyr::select(average_confidence, correlated_satisfactions)
  )
)

# Find the tables

#' Get all the CSV filename
get_all_csv_filenames <- function() {
  list.files(path = "..", pattern = "csv", recursive = TRUE, full.names = TRUE)
}
testthat::expect_true(all(file.exists(get_all_csv_filenames())))

#' Get all the relevant CSV filename
get_csv_filenames <- function() {
  filenames <- get_all_csv_filenames()

  # Remove the helper CSVs
  filenames <- stringr::str_subset(filenames, "average_confidences", negate = TRUE)
  filenames <- stringr::str_subset(filenames, "survey_start", negate = TRUE)
  filenames <- stringr::str_subset(filenames, "initial_knowledge", negate = TRUE)
  filenames <- stringr::str_subset(filenames, "confidences_course", negate = TRUE)

  # Remove evaluations without confidences
  if (exists("has_confidence")) {
    testthat::expect_false(has_confidence("../20230523/evaluation_20230523.csv"))
  }
  filenames <- stringr::str_subset(filenames, "20230523/evaluation_20230523.csv", negate = TRUE)
  if (exists("has_confidence")) {
    testthat::expect_false(has_confidence("../20231201/evaluation_20231201.csv"))
  }
  filenames <- stringr::str_subset(filenames, "20231201/evaluation_20231201.csv", negate = TRUE)
  filenames
}
testthat::expect_true(all(file.exists(get_csv_filenames())))

#' Check that a file has a column related to the rating
has_satisfaction <- function(csv_filename) {
  t <- readr::read_csv(csv_filename, show_col_types = FALSE)
  col_names <- names(t)
  sum(stringr::str_count(col_names, "how would you rate this training event")) != 0
}
for (filename in get_csv_filenames()) {
  # message(filename)
  testthat::expect_true(has_satisfaction(filename))
}

#' Check that a file has a column related to the rating
has_confidence <- function(csv_filename) {
  t <- readr::read_csv(csv_filename, show_col_types = FALSE)
  col_names <- names(t)
  sum(stringr::str_count(col_names, "I can")) != 0
}
testthat::expect_false(has_confidence("../20230523/evaluation_20230523.csv"))
testthat::expect_false(has_confidence("../20231201/evaluation_20231201.csv"))
for (csv_filename in get_csv_filenames()) {
  # message(csv_filename)
  testthat::expect_true(has_confidence(csv_filename))
}

get_satisfactions <- function(csv_filename) {
  testthat::expect_true(has_satisfaction(csv_filename))
  t <- readr::read_csv(csv_filename, show_col_types = FALSE)
  col_name <- stringr::str_subset(names(t), "rate")
  testthat::expect_equal(1, length(col_name))
  satisfactions <- t |> dplyr::select(dplyr::all_of(col_name)) |> tibble::deframe()
  testthat::expect_true(length(satisfactions) > 0)
  satisfactions
}
for (csv_filename in get_csv_filenames()) {
  message(csv_filename, ": ", paste(get_satisfactions(csv_filename), collapse = " "))
  testthat::expect_true(all(get_satisfaction(csv_filename) >= 1.0))
  testthat::expect_true(all(get_satisfaction(csv_filename) <= 10.0))
}

#' Get the average confidences per day
get_average_confidence <- function(csv_filename) {
  testthat::expect_true(has_confidence(csv_filename))
  t <- readr::read_csv(csv_filename, show_col_types = FALSE)
  col_names <- stringr::str_subset(names(t), "I can")

  t_sub <- t |>
    dplyr::select(dplyr::all_of(col_names))

  t_sub <- t_sub |>
    dplyr::mutate_all(~ replace(., . == "I can absolutely do this!", 5)) |>
    dplyr::mutate_all(~ replace(., . == "I have good confidence I can do this", 4)) |>
    dplyr::mutate_all(~ replace(., . == "I have some confidence I can do this", 3)) |>
    dplyr::mutate_all(~ replace(., . == "I have low confidence I can do this", 2)) |>
    dplyr::mutate_all(~ replace(., . == "I have no confidence I can do this", 1)) |>
    dplyr::mutate_all(~ replace(., . == "I don't know even what this is about ...?", 0)) |>
    dplyr::mutate_all(~ replace(., . == "I did not attend that session", NA))

  average_confidence <- rep(NA, nrow(t_sub))
  for (i in seq_len(nrow(t_sub))) {
    confidences <- as.numeric(t_sub[i, ])
    confidences <- confidences[ !is.na(confidences) ]
    average_confidence[i] <- mean(confidences)
  }
  #average_confidence <- average_confidence[ !is.na(average_confidence) ]

  testthat::expect_equal(0, sum(is.na(average_confidence)))
  average_confidence
}
for (csv_filename in get_csv_filenames()) {
  message(csv_filename, ": ", paste(get_average_confidence(csv_filename), collapse = " "))
  testthat::expect_true(all(get_average_confidence(csv_filename) >= 1.0))
  testthat::expect_true(all(get_average_confidence(csv_filename) <= 10.0))
}

csv_filenames <- get_csv_filenames()
list_of_tables <- list()
for (i in seq_len(length(csv_filenames))) {
  csv_filename <- csv_filenames[i]
  message(csv_filename)
  satisfactions <- get_satisfactions(csv_filename)
  average_confidences <- get_average_confidence(csv_filename)
  testthat::expect_equal(length(satisfactions), length(average_confidences))
  t <- tibble::tibble(
    satisfaction = satisfactions,
    average_confidence = average_confidences
  )
  message(are_correlated(t))
  list_of_tables[[i]] <- t
}
t <- dplyr::bind_rows(list_of_tables)


results <- correlation::correlation(t)
p_value <- results$p

model <- lm(average_confidence ~ satisfaction, data = t)
r_squared <- summary(model)$r.squared

ggplot2::ggplot(t, ggplot2::aes(x = satisfaction, y = average_confidence)) +
  ggplot2::geom_jitter(width = 0.01, height = 0.01) +
  ggplot2::geom_smooth(method = "lm") +
  ggplot2::labs(
    title = "Correlation between course satisfaction and average confidence",
    caption = paste0(
      "n: ", nrow(t), ", ",
      "p value: ", round(p_value, digits = 5), ", ",
      "R squared: ", round(100.0 * r_squared, digits = 1), "%"
    )
  )
ggplot2::ggsave("correlation.png", width = 7, height = 7)
