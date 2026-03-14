library(tercen)
library(dplyr, warn.conflicts = FALSE)
library(tidyr)
library(MASS)

ctx <- tercenCtx()

# Parameters
prior_method <- ctx$op.value("prior", type = as.character, default = "equal")
cv_fraction  <- ctx$op.value("cv_fraction", type = as.double, default = 0.0)
max_comp     <- ctx$op.value("maxComp", type = as.integer, default = 5L)

# Get all data using select() instead of as.matrix() + cselect()
df <- ctx$select(unlist(list(".ci", ".ri", ".y", ctx$colors)))

# Build matrix from .ci (observation), .ri (variable), .y (value)
mat_df <- df %>% select(.ci, .ri, .y) %>%
  tidyr::pivot_wider(names_from = .ri, values_from = .y)
mat_obs <- mat_df %>% select(-.ci) %>% as.matrix()

n_obs  <- nrow(mat_obs)
n_vars <- ncol(mat_obs)

cat(paste("DEBUG: Matrix", n_obs, "x", n_vars, "\n"), file = stderr())

# Group labels from color factors (one row per observation)
group_df <- df %>%
  select(.ci, all_of(ctx$colors)) %>%
  distinct() %>%
  arrange(.ci)

if (length(ctx$colors) > 1) {
  group_labels <- apply(group_df[, ctx$colors, drop = FALSE], 1, paste, collapse = ":")
} else {
  group_labels <- group_df[[ctx$colors[1]]]
}
group_labels <- as.factor(group_labels)

cat(paste("DEBUG: Groups", nlevels(group_labels), "\n"), file = stderr())

n_groups <- nlevels(group_labels)
n_ld     <- min(as.integer(max_comp), n_groups - 1, n_vars)

# Hold-out cross-validation
holdout_idx <- rep(FALSE, n_obs)
if (cv_fraction > 0) {
  set.seed(42)
  for (grp in levels(group_labels)) {
    grp_idx   <- which(group_labels == grp)
    n_holdout <- max(1, round(length(grp_idx) * cv_fraction))
    holdout_idx[sample(grp_idx, n_holdout)] <- TRUE
  }
}

# Prior probabilities
if (prior_method == "equal") {
  prior_probs <- rep(1 / n_groups, n_groups)
  names(prior_probs) <- levels(group_labels)
} else {
  prior_probs <- table(group_labels[!holdout_idx]) / sum(!holdout_idx)
}

# Fit LDA
train_X     <- mat_obs[!holdout_idx, , drop = FALSE]
train_group <- group_labels[!holdout_idx]
lda_fit     <- MASS::lda(train_X, grouping = train_group, prior = prior_probs)

cat("DEBUG: LDA fit done\n", file = stderr())

# Predict all observations
pred_all <- predict(lda_fit, mat_obs)

# Eigenvalues and variance explained
eigenvalues   <- lda_fit$svd^2
var_explained <- eigenvalues / sum(eigenvalues)

cat(paste("DEBUG: Var explained:", paste(round(var_explained[1:n_ld], 3), collapse=", "), "\n"), file = stderr())

# --- Build output as simple tibble per observation ---
# Following the same pattern as the PCA operator (which works)

# Build output per observation x LD component
out_list <- list()
for (i in 1:n_ld) {
  out_list[[i]] <- tibble(
    .ci = 0:(n_obs - 1),
    LD = paste0("LD", i),
    ld.score = pred_all$x[, i],
    ld.eigen.value = eigenvalues[i],
    var_between_explained = var_explained[i],
    ld.loading = NA_real_,  # loading is per-variable, not per-obs
    predicted_group = as.character(pred_all$class),
    posterior_max = apply(pred_all$posterior, 1, max),
    is_holdout = as.integer(holdout_idx)
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
