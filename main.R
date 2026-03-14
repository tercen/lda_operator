library(tercen)
library(dplyr, warn.conflicts = FALSE)
select <- dplyr::select
library(tidyr)
library(MASS)

ctx <- tercenCtx()

# Parameters
prior_method <- ctx$op.value("prior", type = as.character, default = "equal")
cv_fraction  <- ctx$op.value("cv_fraction", type = as.double, default = 0.0)
max_comp     <- ctx$op.value("maxComp", type = as.integer, default = 5L)

color_names <- unlist(ctx$colors)

# Get all data
df <- ctx$select(c(".ci", ".ri", ".y", color_names))
cat(paste("DEBUG: df rows:", nrow(df), "cols:", ncol(df), "\n"), file = stderr())

# Build matrix: pivot .ri -> columns to get observations x variables
mat_df <- df %>% select(.ci, .ri, .y) %>%
  tidyr::pivot_wider(names_from = .ri, values_from = .y, names_prefix = "V")
mat_obs <- mat_df %>% select(-.ci) %>% as.matrix()

n_obs  <- nrow(mat_obs)
n_vars <- ncol(mat_obs)
cat(paste("DEBUG: Matrix", n_obs, "x", n_vars, "\n"), file = stderr())

# Group labels: one per observation (.ci)
group_df <- df %>%
  select(.ci, all_of(color_names)) %>%
  distinct(.ci, .keep_all = TRUE) %>%
  arrange(.ci)

cat(paste("DEBUG: group_df rows:", nrow(group_df), "\n"), file = stderr())

if (length(color_names) > 1) {
  group_labels <- apply(group_df[, color_names, drop = FALSE], 1, paste, collapse = ":")
} else {
  group_labels <- group_df[[color_names[1]]]
}
group_labels <- as.factor(group_labels)

n_groups <- nlevels(group_labels)
n_ld     <- min(as.integer(max_comp), n_groups - 1, n_vars)
cat(paste("DEBUG: Groups:", n_groups, "levels:", paste(levels(group_labels), collapse=", "), "\n"), file = stderr())
cat(paste("DEBUG: Group sizes:", paste(table(group_labels), collapse=", "), "\n"), file = stderr())

# Hold-out cross-validation
holdout_idx <- rep(FALSE, n_obs)

# Prior probabilities
if (prior_method == "equal") {
  prior_probs <- rep(1 / n_groups, n_groups)
  names(prior_probs) <- levels(group_labels)
} else {
  prior_probs <- table(group_labels[!holdout_idx]) / sum(!holdout_idx)
}
cat(paste("DEBUG: Priors:", paste(round(prior_probs, 3), collapse=", "), "\n"), file = stderr())

# Fit LDA
lda_fit <- MASS::lda(mat_obs, grouping = group_labels, prior = prior_probs)
cat("DEBUG: LDA fit done\n", file = stderr())

# Predict all observations
pred_all <- predict(lda_fit, mat_obs)

# Eigenvalues and variance explained
eigenvalues   <- lda_fit$svd^2
var_explained <- eigenvalues / sum(eigenvalues)
cat(paste("DEBUG: Var explained:", paste(round(var_explained[1:n_ld], 4), collapse=", "), "\n"), file = stderr())

# Build simple output per observation x LD
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
