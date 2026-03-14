library(tercen)
library(dplyr, warn.conflicts = FALSE)
select <- dplyr::select
library(tidyr)
library(MASS)

ctx <- tercenCtx()

prior_method <- ctx$op.value("prior", type = as.character, default = "equal")
cv_fraction  <- ctx$op.value("cv_fraction", type = as.double, default = 0.0)
max_comp     <- ctx$op.value("maxComp", type = as.integer, default = 5L)

color_names <- unlist(ctx$colors)

df <- ctx$select(c(".ci", ".ri", ".y", color_names))

# Build matrix
mat_df <- df %>% select(.ci, .ri, .y) %>%
  tidyr::pivot_wider(names_from = .ri, values_from = .y, names_prefix = "V")
mat_obs <- mat_df %>% select(-.ci) %>% as.matrix()

cat(paste("DEBUG: Matrix dim:", nrow(mat_obs), ncol(mat_obs), "\n"), file = stderr())
cat(paste("DEBUG: Matrix colnames:", paste(colnames(mat_obs), collapse=", "), "\n"), file = stderr())
cat(paste("DEBUG: Any NAs:", any(is.na(mat_obs)), "\n"), file = stderr())
cat(paste("DEBUG: Matrix class:", class(mat_obs), "\n"), file = stderr())
cat(paste("DEBUG: First row:", paste(mat_obs[1,], collapse=", "), "\n"), file = stderr())

# Group labels
group_df <- df %>%
  select(.ci, all_of(color_names)) %>%
  distinct(.ci, .keep_all = TRUE) %>%
  arrange(.ci)

if (length(color_names) > 1) {
  group_labels <- apply(group_df[, color_names, drop = FALSE], 1, paste, collapse = ":")
} else {
  group_labels <- group_df[[color_names[1]]]
}
group_labels <- as.factor(group_labels)

n_obs  <- nrow(mat_obs)
n_vars <- ncol(mat_obs)
n_groups <- nlevels(group_labels)
n_ld     <- min(as.integer(max_comp), n_groups - 1, n_vars)

# Use simpler column names
colnames(mat_obs) <- paste0("V", 1:ncol(mat_obs))

# Prior
prior_probs <- rep(1 / n_groups, n_groups)
names(prior_probs) <- levels(group_labels)

cat("DEBUG: About to call lda()...\n", file = stderr())

tryCatch({
  lda_fit <- MASS::lda(mat_obs, grouping = group_labels, prior = prior_probs)
  cat("DEBUG: LDA fit done\n", file = stderr())
}, error = function(e) {
  cat(paste("DEBUG: LDA error:", conditionMessage(e), "\n"), file = stderr())
  cat(paste("DEBUG: mat_obs storage.mode:", storage.mode(mat_obs), "\n"), file = stderr())
  cat(paste("DEBUG: typeof mat_obs:", typeof(mat_obs), "\n"), file = stderr())
  # Try with explicit numeric conversion
  mat_obs2 <- matrix(as.numeric(mat_obs), nrow = nrow(mat_obs), ncol = ncol(mat_obs))
  colnames(mat_obs2) <- paste0("V", 1:ncol(mat_obs2))
  cat("DEBUG: Retrying with explicit numeric...\n", file = stderr())
  lda_fit <<- MASS::lda(mat_obs2, grouping = group_labels, prior = prior_probs)
  cat("DEBUG: LDA retry succeeded!\n", file = stderr())
})

# Predict
pred_all <- predict(lda_fit, mat_obs)

eigenvalues   <- lda_fit$svd^2
var_explained <- eigenvalues / sum(eigenvalues)
cat(paste("DEBUG: Var explained:", paste(round(var_explained[1:n_ld], 4), collapse=", "), "\n"), file = stderr())

# Build output
out_list <- list()
for (i in 1:n_ld) {
  out_list[[i]] <- tibble(
    .ci = 0:(n_obs - 1),
    LD = paste0("LD", i),
    ld.score = pred_all$x[, i],
    ld.eigen.value = eigenvalues[i],
    var_between_explained = var_explained[i],
    predicted_group = as.character(pred_all$class),
    posterior_max = apply(pred_all$posterior, 1, max),
    is_holdout = 0L
  )
}

result <- bind_rows(out_list) %>%
  ctx$addNamespace() %>%
  as_relation() %>%
  left_join_relation(ctx$crelation, ".ci", ctx$crelation$rids) %>%
  as_join_operator(ctx$cnames, ctx$cnames)

cat("DEBUG: Saving...\n", file = stderr())
save_relation(result, ctx)
cat("DEBUG: DONE\n", file = stderr())
