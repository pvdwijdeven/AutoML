from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import (
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
)
from sklearn.neural_network import MLPRegressor

# from catboost import CatBoostRegressor

# Example for binary classification (balanced or imbalanced)
models = {
    "binary_classification": {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100),
        "svc": SVC(probability=True),
        "knn": KNeighborsClassifier(n_neighbors=11),
        "gradient_boosting": GradientBoostingClassifier(),
        # "naive_bayes": None,  # Replace with actual model from sklearn.naive_bayes
        # "extra_trees": None,  # Add ExtraTreesClassifier here
        # "adaboost": None,
        # "mlp": None,  # Add MLPClassifier here
        # "linear_discriminant_analysis": None,  # Add LDA here
    },
    "imbalanced_binary_classification": {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100),
        "svc": SVC(probability=True),
        "knn": KNeighborsClassifier(n_neighbors=11),
        "gradient_boosting": GradientBoostingClassifier(),
        # "naive_bayes": None,  # Replace with actual model from sklearn.naive_bayes
        # "extra_trees": None,  # Add ExtraTreesClassifier here
        # "adaboost": None,
        # "mlp": None,  # Add MLPClassifier here
        # "linear_discriminant_analysis": None,  # Add LDA here
    },
    "regression": {
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(alpha=1.0),
        "lasso_regression": Lasso(alpha=0.1),
        "elasticnet_regression": ElasticNet(alpha=0.1, l1_ratio=0.5),
        "bayesian_ridge_regression": BayesianRidge(),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
        "gradient_boosting_regressor": GradientBoostingRegressor(
            n_estimators=100, random_state=42
        ),
        "svr": SVR(kernel="rbf", C=1.0, epsilon=0.1),
        "knn_regressor": KNeighborsRegressor(n_neighbors=5),
        "mlp_regressor": MLPRegressor(
            hidden_layer_sizes=(100,), max_iter=500, random_state=42
        ),
    },
}


regression_models = {
    "linear_regression": LinearRegression(),
    "ridge_regression": Ridge(alpha=1.0),
    "lasso_regression": Lasso(alpha=0.1),
    "elasticnet_regression": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "bayesian_ridge_regression": BayesianRidge(),
    "random_forest_regressor": RandomForestRegressor(
        n_estimators=100, random_state=42
    ),
    "gradient_boosting_regressor": GradientBoostingRegressor(
        n_estimators=100, random_state=42
    ),
    "svr": SVR(kernel="rbf", C=1.0, epsilon=0.1),
    "knn_regressor": KNeighborsRegressor(n_neighbors=5),
    "mlp_regressor": MLPRegressor(
        hidden_layer_sizes=(100,), max_iter=500, random_state=42
    ),
    # "catboost_regressor": CatBoostRegressor(
    #     verbose=0, random_seed=42
    # ),  # suppress CatBoost output
}
