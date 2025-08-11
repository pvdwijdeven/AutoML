### Conceptual Signature of a Preprocessing Step Function

Here is a generalized function signature that fits your requirements:

```python
def preprocess_step_fit_transform(
    X_train: pd.DataFrame,
    y_train: pd.Series = None,
    *,
    fit: bool,
    step_params: dict = None,
    target_aware: bool = False,
    **kwargs
) -> tuple:
    """
    Fit and/or transform the provided dataset using a specific preprocessing step.

    Parameters:
    - X_train: DataFrame of features to be processed.
    - y_train: Optional Series or DataFrame of targets, required for steps affecting targets.
    - fit: Whether to calculate/store parameters from this dataset (True when called on training data).
    - step_params: Previously fitted parameters dict from training step (None if fit=True).
    - target_aware: Boolean flag indicating the step might modify targets or row removals affecting both X and y.
    - **kwargs: Additional step-specific parameters.

    Returns:
    - X_transformed: Processed feature DataFrame.
    - y_transformed: Processed target Series/DataFrame (or None if target_aware=False).
    - step_params_out: Parameters learned or confirmed during this step, saved for reproduction on new data.
    """

    # Example flow inside:
    # if fit:
    #    compute and store necessary parameters in step_params_out
    #    transform X_train (and y_train if target_aware)
    # else:
    #    use step_params to transform X_train (and y_train if needed)
    #
    # return processed data and step_params_out
```

***

### Example Usage for Imputation Step

For an imputation step that fills missing values with the median per column, here is a simplified example outline:

```python
def impute_missing_values(
    X: pd.DataFrame,
    y: pd.Series = None,
    *,
    fit: bool,
    step_params: dict = None,
    target_aware: bool = False,
    columns: list = None
) -> tuple:
    if fit:
        # Calculate median per column from training data
        medians = X[columns].median()
        X_transformed = X.fillna(medians)
        step_params_out = {'medians': medians.to_dict()}
    else:
        medians = pd.Series(step_params['medians'])
        X_transformed = X.fillna(medians)
        step_params_out = step_params

    # For imputation, targets remain unchanged
    if target_aware:
        # In case you want to drop rows or modify targets, handle here (not needed in simple impute)
        y_transformed = y
    else:
        y_transformed = None

    return X_transformed, y_transformed, step_params_out
```

***

### Integrating Into Your Preprocessing Class

Your preprocessing class would:

- Keep a **list of step names and their saved parameters** in order.
- For a new training dataset: call each step with `fit=True`, saving returned step_params.
- For a new validation/test set: call each step with `fit=False` and the saved parameters to replicate identical transformations.
- If a step modifies rows (dropping), ensure both features and targets are synchronized accordingly in outputs.

Example internal state for storing steps might be:

```python
self.steps = [
    {"name": "impute_missing", "params": None, "function": impute_missing_values, "target_aware": False},
    {"name": "drop_missing_targets", "params": None, "function": drop_missing_targets, "target_aware": True},
    # more steps...
]
```

When fit on train:

```python
for step in self.steps:
    X_train, y_train, step['params'] = step['function'](X_train, y_train, fit=True, target_aware=step['target_aware'], **step.get('config', {}))
```

When transform on val/test:

```python
for step in self.steps:
    X_val, y_val, _ = step['function'](X_val, y_val, fit=False, step_params=step['params'], target_aware=step['target_aware'], **step.get('config', {}))
```

Example of config in step:

```python
self.steps = [
    {
        "name": "impute_missing",
        "params": None,
        "function": impute_missing_values,
        "target_aware": False,
        "config": {
            "columns": ["age", "salary"]   # columns to impute
        }
    },
    {
        "name": "drop_missing_targets",
        "params": None,
        "function": drop_missing_targets,
        "target_aware": True,
        # no config provided here
    }
]
```
***
