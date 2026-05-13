#!/bin/env Rscript

################
# Confidences
################
get_confidences_files <- function() {
  confidences_files <- tibble::tribble(
    ~course, ~day, ~confidences_filename,
    7, 1, "../../../naiss_intro_python/docs/evaluations/20250424/average_confidences.csv",
    7, 2, "../20250425_day_2/average_confidences.csv",
    7, 3, "../20250428_day_3/average_confidences.csv",
    7, 4, "../20250429_day_4/average_confidences.csv",
    8, 1, "../../../naiss_intro_python/docs/evaluations/20251127/average_confidences.csv",
    8, 2, "../20251128_day_2/average_confidences.csv",
    8, 3, "../20251201_day_3/average_confidences.csv",
    8, 4, "../20251202_day_4/average_confidences.csv",
    9, 1, "../../../naiss_intro_python/docs/evaluations/20260420/average_confidences.csv",
    9, 2, "../20260422_day_2/average_confidences.csv",
    9, 3, "../20260423_day_3/average_confidences.csv",
    9, 4, "../20260424_day_4/average_confidences.csv"
  )
  testthat::expect_all_true(file.exists(confidences_files$confidences_filename))
  confidences_files
}
get_confidences_files()

read_confidences <- function(course, day) {
  confidences_files <- get_confidences_files()
  filename <- confidences_files[confidences_files$course == course & confidences_files$day == day, ]$confidences_filename
  testthat::expect_true(file.exists(filename))
  t <- readr::read_csv(filename, show_col_types = FALSE)
  t
}

confidences_list <- list()
confidences_files <- get_confidences_files()
n_confidences <- nrow(confidences_files)
for (row_index in seq_len(n_confidences)) {
  course <- confidences_files[row_index, ]$course
  day <-  confidences_files[row_index, ]$day
  t <- read_confidences(course, day)
  t$course <- course
  t$day <- day
  confidences_list[[row_index]] <- t
}
confidences_per_day <- dplyr::bind_rows(confidences_list)

ggplot2::ggplot(confidences_per_day,
  ggplot2::aes(
    x = mean
  )
) +
  ggplot2::geom_histogram(binwidth = 0.5) +
  ggplot2::scale_x_continuous(
    limits = c(0, NA),
    breaks = seq(from = 0, to = 5),
    minor_breaks = seq(from = 0, to = 5.25, by = 0.25)
  ) +
  ggplot2::facet_grid(rows = ggplot2::vars(day), cols = ggplot2::vars(course)) +
  ggplot2::theme(
    strip.text.y = ggplot2::element_text(angle = 0),
    legend.position = "bottom",
    axis.text = ggplot2::element_text(size = 7)
  ) +
  ggplot2::labs(
    title = "Average confidences per question, per course, per day",
      caption = "Columns: course number. Rows: day within that course"
    )
  )

ggplot2::ggsave(filename = "confidences_per_course_per_day.png", width = 7, height = 7)

################
# Feedback
################
get_comment_files <- function() {
  comment_files <- tibble::tribble(
    ~course, ~day, ~comment_filename,
    7, 1, "../../../naiss_intro_python/docs/evaluations/20250424/survey_end_text_question.txt",
    7, 2, "../20250425_day_2/comments.txt",
    7, 3, "../20250428_day_3/comments.txt",
    7, 4, "../20250429_day_4/comments.txt",
    8, 1, "../../../naiss_intro_python/docs/evaluations/20251127/survey_end_text_question.txt",
    8, 2, "../20251128_day_2/comments.txt",
    8, 3, "../20251201_day_3/comments.txt",
    8, 4, "../20251202_day_4/comments.txt",

    9, 1, "../../../naiss_intro_python/docs/evaluations/20260420/survey_end_text_question.txt",
    9, 2, "../20260422_day_2/comments.txt",
    9, 3, "../20260423_day_3/comments.txt",
    9, 4, "../20260424_day_4/comments.txt"
  )
  testthat::expect_all_true(file.exists(comment_files$comment_filename))
  comment_files
}
get_comment_files()

read_comments <- function(course, day) {
  comment_files <- get_comment_files()
  filename <- comment_files[comment_files$course == course & comment_files$day == day, ]$comment_filename
  testthat::expect_true(file.exists(filename))
  lines <- readr::read_lines(filename)
  lines <- lines[!is.na(lines)]
  lines <- stringr::str_subset(lines, pattern = "NA|N/a|________", negate = TRUE)
  lines[lines != ""]
}

comments_list <- list()
comment_files <- get_comment_files()
n_comments <- nrow(comment_files)
for (row_index in seq_len(n_comments)) {
  course <- comment_files[row_index, ]$course
  day <-  comment_files[row_index, ]$day
  comments_list[[row_index]] <- tibble::tibble(
    course = course,
    day = day,
    comment = read_comments(course, day)
  )
}
comments_per_day <- dplyr::bind_rows(comments_list)
comments_per_day$comment

readr::write_csv(comments_per_day, "comments_per_course_per_day.csv")
text <- knitr::kable(comments_per_day)
readr::write_lines(text, "comments_per_course_per_day.md")
