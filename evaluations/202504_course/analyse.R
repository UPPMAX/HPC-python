#!/bin/env Rscript

read_day_1_confidences <- function() {
  readr::read_csv("../../../naiss_intro_python/docs/evaluations/20250424/average_confidences.csv", show_col_types = FALSE)
}
read_day_2_confidences <- function() {
  readr::read_csv("../20250425_day_2/average_confidences.csv", show_col_types = FALSE)
}
read_day_3_confidences <- function() {
  readr::read_csv("../20250428_day_3/average_confidences.csv", show_col_types = FALSE)
}
read_day_4_confidences <- function() {
  readr::read_csv("../20250429_day_4/average_confidences.csv", show_col_types = FALSE)
}

get_day_1_condidences <- function() {
  t <- read_day_1_confidences()
  names(t) <- c("learning_outcomes", "average_confidence")
  t$day <- 1
  t
}
get_day_2_condidences <- function() {
  t <- read_day_2_confidences()
  names(t) <- c("learning_outcomes", "average_confidence")
  t$day <- 2
  t
}
get_day_3_condidences <- function() {
  t <- read_day_3_confidences()
  names(t) <- c("learning_outcomes", "average_confidence")
  t$day <- 3
  t  
}
get_day_4_condidences <- function() {
  t <- read_day_4_confidences()
  names(t) <- c("learning_outcomes", "average_confidence")
  t$day <- 4
  t  
}
get_condidences <- function() {
  dplyr::bind_rows(
    get_day_1_condidences(), 
    get_day_2_condidences(), 
    get_day_3_condidences(), 
    get_day_4_condidences()
  )
}

t <- get_condidences()
t$day <- as.factor(t$day)
t <- t |> dplyr::arrange(average_confidence)
t$learning_outcomes <- as.factor(t$learning_outcomes)
t$learning_outcomes <- reorder(
  x = t$learning_outcomes, 
  X = order(t$average_confidence),
  decreasing = TRUE
)

average_average_confidence <- mean(t$average_confidence)

ggplot2::ggplot(t, 
  ggplot2::aes(
    x = average_confidence, 
    y = learning_outcomes,
    fill = day
  )
) +
  ggplot2::geom_col() +
  ggplot2::geom_vline(xintercept = average_average_confidence, lty = "dashed") +
  ggplot2::theme(
    strip.text.y = ggplot2::element_text(angle = 0),
    legend.position = "bottom"
  ) +
  ggplot2::labs(
    title = "Confidences per question",
    caption = paste0(
      " Dashed line denotes the average at ", round(average_average_confidence, digits = 2)
    )
  )

ggplot2::ggsave(filename = "confidences_per_question.png", width = 7, height = 7)
