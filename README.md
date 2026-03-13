# LDA Operator

Performs Fisher's Linear Discriminant Analysis (LDA) / Canonical Discriminant Analysis.

## Input

| Zone   | Description              |
|--------|--------------------------|
| Row    | Variable identifiers     |
| Column | Observation identifiers  |
| Y-axis | Measurement values       |
| Color  | Group factor (required)  |

## Output

| Attribute              | Type   | Description                                          |
|------------------------|--------|------------------------------------------------------|
| LD                     | string | Canonical variate label (LD1, LD2, ...)              |
| ld.score               | double | Discriminant score per observation per variate        |
| ld.loading             | double | Scaling coefficient per variable per variate          |
| ld.eigen.value         | double | Eigenvalue for this variate                           |
| var_between_explained  | double | Proportion of between-group variance explained        |
| predicted_group        | string | Predicted class from posterior probabilities           |
| posterior_max          | double | Maximum posterior probability (confidence)             |
| is_holdout             | int32  | 1 if held out for cross-validation, 0 otherwise       |

## Properties

| Name        | Default       | Description                                              |
|-------------|---------------|----------------------------------------------------------|
| prior       | equal         | Prior probabilities: "equal" or "proportional"           |
| cv_fraction | 0.0           | Fraction held out per group for cross-validation         |
| maxComp     | 5             | Maximum number of canonical variates to return           |
