t_wide <- readr::read_csv("cluster_usage.csv")

n_clusters <- ncol(t_wide)
t_wide$id <- seq(1, nrow(t_wide))

t <- t_wide |> tidyr::pivot_longer(cols = 1:n_clusters)

names(t) <- c("id", "hpc_cluster", "do_use")
t$do_use[is.na(t$do_use)] <- "no"
t$do_use[t$do_use == "X"] <- "yes"

n_per_hpc_cluster <- t |> dplyr::filter(do_use == "yes") |>
  dplyr::select(hpc_cluster) |>
  dplyr::group_by(hpc_cluster) |>
  dplyr::tally()

knitr::kable(n_per_hpc_cluster)

