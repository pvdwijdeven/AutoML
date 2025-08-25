# Third-party imports
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBRegressor


# Example for binary classification (balanced or imbalanced)
models = {
    "binary_classification": {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=500),
        "svc": SVC(probability=True),
        "knn": KNeighborsClassifier(n_neighbors=11),
        "gradient_boosting": GradientBoostingClassifier(),
        "naive_bayes": GaussianNB(),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=100
        ),  # Default 100 trees
        "adaboost": AdaBoostClassifier(),
        "mlp": MLPClassifier(max_iter=1000),
        "linear_discriminant_analysis": LinearDiscriminantAnalysis(),
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
        "xgboost_regressor": XGBRegressor(n_estimators=100, random_state=42),
    },
}


model_class_map = {
    "logistic_regression": LogisticRegression(),
    "random_forest": RandomForestClassifier(),
    "svc": SVC(),
    "knn": KNeighborsClassifier(),
    "gradient_boosting": GradientBoostingClassifier(),
    "naive_bayes": GaussianNB(),
    "extra_trees": ExtraTreesClassifier(),
    "adaboost": AdaBoostClassifier(),
    "mlp": MLPClassifier(),
    "linear_discriminant_analysis": LinearDiscriminantAnalysis(),
    "linear_regression": LinearRegression(),
    "ridge_regression": Ridge(),
    "lasso_regression": Lasso(),
    "elasticnet_regression": ElasticNet(),
    "bayesian_ridge_regression": BayesianRidge(),
    "random_forest_regressor": RandomForestRegressor(),
    "gradient_boosting_regressor": GradientBoostingRegressor(),
    "svr": SVR(),
    "knn_regressor": KNeighborsRegressor(),
    "mlp_regressor": MLPRegressor(),
    "xgboost_regressor": XGBRegressor(),
}

models_family = {
    "binary_classification": {
        "logistic_regression": "Linear Model",
        "random_forest": "Tree Ensemble (Bagging)",
        "svc": "Kernel-Based/Classical ML",
        "knn": "Instance-Based/Lazy Learning",
        "gradient_boosting": "Tree Ensemble (Boosting)",
        "naive_bayes": "Probabilistic",
        "extra_trees": "Tree Ensemble (Bagging)",
        "adaboost": "Tree Ensemble (Boosting)",
        "mlp": "Neural Network",
        "linear_discriminant_analysis": "Linear Model",
    },
    "imbalanced_binary_classification": {
        "logistic_regression": "Linear Model",
        "random_forest": "Tree Ensemble (Bagging)",
        "svc": "Kernel-Based/Classical ML",
        "knn": "Instance-Based/Lazy Learning",
        "gradient_boosting": "Tree Ensemble (Boosting)",
        # Add others with their family/type when included
    },
    "regression": {
        "linear_regression": "Linear Model",
        "ridge_regression": "Regularized Linear Model",
        "lasso_regression": "Regularized Linear Model",
        "elasticnet_regression": "Regularized Linear Model",
        "bayesian_ridge_regression": "Bayesian Linear Model",
        "random_forest_regressor": "Tree Ensemble (Bagging)",
        "gradient_boosting_regressor": "Tree Ensemble (Boosting)",
        "svr": "Kernel-Based/Classical ML",
        "knn_regressor": "Instance-Based/Lazy Learning",
        "mlp_regressor": "Neural Network",
        "xgboost_regressor": "Tree Ensemble (Boosting)",
    },
}
