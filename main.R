cat("START\n", file = stderr())
library(tercen)
cat("tercen loaded\n", file = stderr())
tryCatch({
  library(tercenApi)
  cat("tercenApi loaded\n", file = stderr())
}, error = function(e) {
  cat(paste("tercenApi failed:", conditionMessage(e), "\n"), file = stderr())
})
library(dplyr, warn.conflicts = FALSE)
library(tidyr)
library(MASS)
cat("All libraries loaded\n", file = stderr())

cat("Creating context...\n", file = stderr())
ctx <- tercenCtx()
cat("Context created\n", file = stderr())

# Parameters
prior_method <- ctx$op.value("prior", type = as.character, default = "equal")
cv_fraction  <- ctx$op.value("cv_fraction", type = as.double, default = 0.0)
max_comp     <- ctx$op.value("maxComp", type = as.integer, default = 5L)
cat("Parameters loaded\n", file = stderr())

# Data matrix: variables (rows) x observations (cols)
mat <- ctx$as.matrix()
X   <- t(mat)  # observations x variables
cat(paste("Matrix loaded:", nrow(mat), "x", ncol(mat), "\n"), file = stderr())

n_obs  <- nrow(X)
n_vars <- ncol(X)

# Group labels from color factor
if (length(ctx$colors) < 1) stop("A color factor is required for LDA grouping.")

group_df <- ctx$cselect(ctx$colors)
if (ncol(group_df) > 1) {
  group_labels <- apply(group_df, 1, paste, collapse = ":")
} else {
  group_labels <- group_df[[1]]
}
group_labels <- as.factor(group_labels)
cat(paste("Groups:", nlevels(group_labels), "levels\n"), file = stderr())

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
train_X     <- X[!holdout_idx, , drop = FALSE]
train_group <- group_labels[!holdout_idx]
lda_fit     <- MASS::lda(train_X, grouping = train_group, prior = prior_probs)
cat("LDA fit complete\n", file = stderr())

# Predict all observations
pred_all <- predict(lda_fit, X)
cat("Predictions complete\n", file = stderr())

# Eigenvalues and variance explained
eigenvalues   <- lda_fit$svd^2
var_explained <- eigenvalues / sum(eigenvalues)
cat(paste("Variance explained:", paste(round(var_explained, 3), collapse = ", "), "\n"), file = stderr())

# --- Build relations (following PCA operator pattern) ---

# LD component relation (capped at n_ld)
ldRelation <- tibble(
  LD = sprintf(paste0("LD%0", nchar(as.character(n_ld)), "d"), 1:n_ld)
) %>%
  ctx$addNamespace() %>%
  as_relation()
cat("LD relation built\n", file = stderr())

# Eigenvalue relation (one row per LD)
eigenRelation <- tibble(
  ld.eigen.value        = eigenvalues[1:n_ld],
  var_between_explained = var_explained[1:n_ld]
) %>%
  ctx$addNamespace() %>%
  as_relation()
cat("Eigen relation built\n", file = stderr())

# Loadings relation (n_vars x n_ld, pivoted long)
loadingRelation <- lda_fit$scaling[, 1:n_ld, drop = FALSE] %>%
  as_tibble() %>%
  setNames(0:(ncol(.) - 1)) %>%
  mutate(.var.rids = 0:(nrow(.) - 1)) %>%
  pivot_longer(
    -.var.rids,
    names_to        = ".ld.rids",
    values_to       = "ld.loading",
    names_transform = list(.ld.rids = as.integer)
  ) %>%
  ctx$addNamespace() %>%
  as_relation() %>%
  left_join_relation(ctx$rrelation, ".var.rids", ctx$rrelation$rids)
cat("Loading relation built\n", file = stderr())

# Scores relation with classification data (n_obs x n_ld, pivoted long)
scores_wide <- pred_all$x[, 1:n_ld, drop = FALSE] %>%
  as_tibble() %>%
  setNames(0:(ncol(.) - 1))

scores_wide$.i              <- 0:(n_obs - 1)
scores_wide$predicted_group <- as.character(pred_all$class)
scores_wide$posterior_max   <- apply(pred_all$posterior, 1, max)
scores_wide$is_holdout      <- as.integer(holdout_idx)

scoresRelation <- scores_wide %>%
  pivot_longer(
    cols            = all_of(as.character(0:(n_ld - 1))),
    names_to        = ".ld.rids",
    values_to       = "ld.score",
    names_transform = list(.ld.rids = as.integer)
  ) %>%
  ctx$addNamespace() %>%
  as_relation() %>%
  left_join_relation(ctx$crelation, ".i", ctx$crelation$rids)
cat("Scores relation built\n", file = stderr())

# Combine into single join operator
rels <- ldRelation %>%
  left_join_relation(scoresRelation, ldRelation$rids, ".ld.rids") %>%
  left_join_relation(eigenRelation, ldRelation$rids, eigenRelation$rids) %>%
  left_join_relation(loadingRelation, ldRelation$rids, ".ld.rids") %>%
  as_join_operator(ctx$cnames, ctx$cnames)
cat("Join operator built\n", file = stderr())

cat("Saving...\n", file = stderr())
save_relation(rels, ctx)
cat("DONE\n", file = stderr())
