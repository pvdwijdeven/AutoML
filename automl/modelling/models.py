from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.multioutput import MultiOutputClassifier


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
    }
}
