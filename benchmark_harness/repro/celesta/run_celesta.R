# Run unmodified CELESTA (installed package) per region. Called by celesta_repro.py inside
# the `celesta` conda env:
#   Rscript run_celesta.R imaging=<csv|builtin> out=<dir> region_col=<col> regions=<a,b|all>
#           filter_high=<0.9> filter_low=<0.5|none> number_of_neighbors=5 bandwidth=100
# prior matrix and the four threshold vectors are the package's shipped ones (reg009).
suppressPackageStartupMessages({
  library(CELESTA); library(Rmixmod); library(spdep); library(ggplot2); library(reshape2); library(zeallot)
})
kv <- strsplit(commandArgs(trailingOnly = TRUE), "=", fixed = TRUE)
a <- setNames(lapply(kv, function(x) paste(x[-1], collapse = "=")), sapply(kv, `[`, 1))
get <- function(k, d) if (is.null(a[[k]])) d else a[[k]]
data(prior_marker_info); data(imaging_data)
data(high_marker_threshold_anchor); data(low_marker_threshold_anchor)
data(high_marker_threshold_iteration); data(low_marker_threshold_iteration)

out <- normalizePath(get("out", "."), mustWork = FALSE); dir.create(out, recursive = TRUE, showWarnings = FALSE)
if (get("imaging", "builtin") == "builtin") {
  tab <- imaging_data
} else {
  tab <- read.csv(a$imaging, check.names = FALSE)
  names(tab)[names(tab) == "X:X"] <- "X"; names(tab)[names(tab) == "Y:Y"] <- "Y"
}
region_col <- get("region_col", "File Name")
regions <- get("regions", "all")
regions <- if (regions == "all") sort(unique(tab[[region_col]])) else strsplit(regions, ",")[[1]]
fh <- as.numeric(get("filter_high", "0.9")); fl <- get("filter_low", "0.5")
setwd(out)
for (r in regions) {
  t0 <- Sys.time()
  sub <- tab[tab[[region_col]] == r, , drop = FALSE]
  meta <- intersect(c("CellID", "ClusterName", region_col, "X", "Y"), names(sub))
  write.csv(sub[, meta], paste0(r, "_input_meta.csv"), row.names = FALSE)  # row order = CELESTA output order
  obj <- CreateCelestaObject(project_title = r, prior_marker_info, sub,
                             number_of_neighbors = as.numeric(get("number_of_neighbors", "5")),
                             bandwidth = as.numeric(get("bandwidth", "100")))
  if (fl != "none") obj <- FilterCells(obj, high_marker_threshold = fh, low_marker_threshold = as.numeric(fl))
  obj <- AssignCells(obj, max_iteration = 10, cell_change_threshold = 0.01,
                     high_expression_threshold_anchor = high_marker_threshold_anchor,
                     low_expression_threshold_anchor = low_marker_threshold_anchor,
                     high_expression_threshold_index = high_marker_threshold_iteration,
                     low_expression_threshold_index = low_marker_threshold_iteration)
  cat(sprintf("[celesta] %s: %d cells, %.1f s\n", r, nrow(sub), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
}
