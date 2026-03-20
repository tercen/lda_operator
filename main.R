library(tercen)
library(tercenApi)
library(dplyr, warn.conflicts = FALSE)
library(tidyr)
library(MASS)

ctx <- tercenCtx()

# ---- Parameters ----
prior_method <- ctx$op.value("prior", type = as.character, default = "equal")
cv_fraction  <- ctx$op.value("cv_fraction", type = as.double, default = 0.0)
max_comp     <- as.integer(ctx$op.value("maxComp", type = as.double, default = 5))

# ---- Get all data via ctx$select() ----
# Colors are axis-level, not column-level, so we use ctx$select() (NOT ctx$cselect())
df <- ctx$select(unlist(list(".ci", ".ri", ".y", ctx$colors)))

# ---- Build matrix from .ci (observation) x .ri (variable) ----
mat_df <- df %>%
  select(.ci, .ri, .y) %>%
  pivot_wider(names_from = .ri, values_from = .y, values_fn = list(.y = mean))

mat_obs <- mat_df %>% select(-.ci) %>% as.matrix()  # observations x variables

n_obs  <- nrow(mat_obs)
n_vars <- ncol(mat_obs)

# ---- Group labels (one per observation, from color factor) ----
group_df <- df %>%
  select(.ci, all_of(ctx$colors)) %>%
  distinct() %>%
  arrange(.ci)

# Combine color factors into single group label if multiple
if (length(ctx$colors) > 1) {
  group_labels <- as.factor(do.call(paste, c(group_df[ctx$colors], sep = ".")))
} else {
  group_labels <- as.factor(group_df[[ctx$colors[1]]])
}

n_groups <- nlevels(group_labels)
n_ld     <- min(max_comp, n_groups - 1, n_vars)

# ---- Cross-validation hold-out ----
holdout_idx <- rep(FALSE, n_obs)
if (cv_fraction > 0) {
  set.seed(42)
  for (grp in levels(group_labels)) {
    grp_idx <- which(group_labels == grp)
    n_holdout <- max(1, round(length(grp_idx) * cv_fraction))
    holdout_idx[sample(grp_idx, n_holdout)] <- TRUE
  }
}

# ---- Prior probabilities ----
if (prior_method == "equal") {
  prior_probs <- rep(1 / n_groups, n_groups)
  names(prior_probs) <- levels(group_labels)
} else {
  prior_probs <- table(group_labels[!holdout_idx]) / sum(!holdout_idx)
}

# ---- Fit LDA on training data ----
train_X     <- mat_obs[!holdout_idx, , drop = FALSE]
train_group <- group_labels[!holdout_idx]

lda_fit <- MASS::lda(train_X, grouping = train_group, prior = prior_probs)

# ---- Predict all observations ----
pred_all <- predict(lda_fit, mat_obs)

# ---- Extract components ----
scores <- pred_all$x[, 1:n_ld, drop = FALSE]             # n_obs x n_ld
eigenvalues <- lda_fit$svd^2
var_explained <- eigenvalues / sum(eigenvalues)
loadings_mat <- lda_fit$scaling[, 1:n_ld, drop = FALSE]  # n_vars x n_ld
predicted_group <- as.character(pred_all$class)
posterior_max <- apply(pred_all$posterior, 1, max)

nld <- n_ld
nchar_ld <- nchar(as.character(nld))

# ---- Build composite relation (following PCA operator pattern) ----

# 1. LD label relation (the pivot dimension)
ldRelation <- tibble(LD = sprintf(paste0("LD%0", nchar_ld, "d"), 1:nld)) %>%
  ctx$addNamespace() %>%
  as_relation()

# 2. Eigenvalue / variance-explained relation (one per LD)
eigenRelation <- tibble(
  ld.eigen.value = eigenvalues[1:nld],
  var_between_explained = var_explained[1:nld]
) %>%
  ctx$addNamespace() %>%
  as_relation()

# 3. Loading relation (n_vars x n_ld, joined to row relation)
loadingRelation <- loadings_mat %>%
  as.data.frame() %>%
  as_tibble() %>%
  setNames(0:(nld - 1)) %>%
  mutate(.var.rids = 0:(n_vars - 1)) %>%
  pivot_longer(
    -.var.rids,
    names_to = ".ld.rids",
    values_to = "ld.loading",
    names_transform = list(.ld.rids = as.integer)
  ) %>%
  ctx$addNamespace() %>%
  as_relation() %>%
  left_join_relation(ctx$rrelation, ".ld.rids", ctx$rrelation$rids)

# 4. Scores + classification relation (n_obs x n_ld, joined to column relation)
# Classification columns (predicted_group, posterior_max, is_holdout) are per-observation
# and get repeated for each LD after pivot_longer — correct output shape.
scores_df <- scores %>%
  as.data.frame() %>%
  as_tibble() %>%
  setNames(0:(nld - 1)) %>%
  mutate(
    .i = 0:(n_obs - 1),
    predicted_group = predicted_group,
    posterior_max = posterior_max,
    is_holdout = as.integer(holdout_idx)
  ) %>%
  pivot_longer(
    cols = all_of(as.character(0:(nld - 1))),
    names_to = ".ld.rids",
    values_to = "ld.score",
    names_transform = list(.ld.rids = as.integer)
  )

scoresRelation <- scores_df %>%
  ctx$addNamespace() %>%
  as_relation() %>%
  left_join_relation(ctx$crelation, ".i", ctx$crelation$rids)

# ---- Link all relations into one composite ----
rels <- ldRelation %>%
  left_join_relation(scoresRelation, ldRelation$rids, ".ld.rids") %>%
  left_join_relation(eigenRelation, ldRelation$rids, eigenRelation$rids) %>%
  left_join_relation(loadingRelation, ldRelation$rids, ".ld.rids") %>%
  as_join_operator(ctx$cnames, ctx$cnames)

save_relation(rels, ctx)
