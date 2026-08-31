#!/bin/env Rscript
prevaluation_file <- "survey_start.csv"

testthat::expect_true(file.exists(prevaluation_file))

t_pre_raw <- readr::read_csv(prevaluation_file, show_col_types = FALSE)

t_pre_raw$Timestamp <- NULL

#' Shorten the names of the columns
shorten_col_names <- function(t) {
  questions <- stringr::str_remove(
    stringr::str_remove(
      names(t),
      "Give you confidence levels of the following statements below: \\["),
    "\\]"
  )

  names(t) <- questions
  t
}

#' Convert the table to tidy format
#' Add a columns 'i' for the index of an individual
tidy_table <- function(t = t_pre) {
  t$i <- seq(1, nrow(t))
  t_tidy <- tidyr::pivot_longer(t, cols = starts_with("I", ignore.case = FALSE))
  names(t_tidy) <- c("i", "question", "answer")
  t_tidy
}

t_pre_untidy <- shorten_col_names(t_pre_raw)

t_pre <- tidy_table(t_pre_untidy)
t_pre$when <- "pre"

plot_histrogram <- function(t_tidy) {

  n_individuals <- length(unique(t_tidy$i))
  n_ratings <- length(t_tidy$answer[!is.na(t_tidy$answer)])
  mean_confidence <- mean(t_tidy$answer[!is.na(t_tidy$answer)])

  ggplot2::ggplot(t_tidy, ggplot2::aes(x = answer)) +
    ggplot2::geom_density() +
    ggplot2::labs(
      title = "All confidences",
      caption = paste0(
        "#individuals: ", n_individuals, ". ",
        "#ratings: ", n_ratings, ". ",
        "Mean confidence: ", round(mean_confidence, digits = 2)
      )
    )

}

plot_histrogram(t_pre)

mean_pre <- mean(t_pre$answer, na.rm = TRUE)

ggplot2::ggplot(t_pre, ggplot2::aes(x = answer)) +
  ggplot2::geom_density(alpha = 0.5) +
  ggplot2::geom_vline(xintercept = mean_pre, color = "skyblue", lty = "dashed") +
  ggplot2::labs(
    title = "All confidences",
    caption = paste0(
      "Mean pre: ", format(mean_pre, digits = 2)
    )
  )

ggplot2::ggsave(filename = "all_confidences_pre.png", width = 6, height = 2)

ggplot2::ggplot(t_pre, ggplot2::aes(x = answer)) +
  ggplot2::geom_density(alpha = 0.5) +
  ggplot2::facet_grid(rows = "question", scales = "free_y") +
  ggplot2::theme(
    strip.text.y = ggplot2::element_text(angle = 0),
    legend.position = "none"
  ) +
  ggplot2::labs(
    title = "Confidences per question"
  )

ggplot2::ggsave(filename = "confidences_per_question_pre.png", width = 6, height = 7)

names(t)

ggplot2::ggplot(
  t_pre,
  ggplot2::aes(x = question, y = answer)) +
  ggplot2::geom_boxplot(position = "dodge") +
  ggplot2::theme(
    axis.text.x = ggplot2::element_text(angle = 90, vjust = 0.5, hjust = 1),
    legend.position = "none"
  ) +
  ggplot2::labs(
    title = "Confidences per question"
  )
ggplot2::ggsave(filename = "confidences_per_question_boxplot_pre.png", width = 6, height = 7)

# Get the average
t_averages <- t_pre |> dplyr::group_by(question) |> dplyr::summarise(mean = mean(answer))
ggplot2::ggplot(
  t_averages,
  ggplot2::aes(x = question, y = mean)) +
  ggplot2::geom_col(position = "dodge") +
  ggplot2::theme(
    axis.text.x = ggplot2::element_text(angle = 90, vjust = 0.5, hjust = 1),
    legend.position = "none"
  ) +
  ggplot2::labs(
    title = "Confidences per question"
  )
ggplot2::ggsave(filename = "average_confidences_per_question_pre.png", width = 6, height = 7)
