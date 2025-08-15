import numpy as np
from sklearn.tree import DecisionTreeClassifier

param_grids = {
    "binary_classification": {
        "LogisticRegression": {
            "penalty": ["l1", "l2", "elasticnet", "none"],
            "C": np.logspace(-4, 4, 20),
            "solver": ["lbfgs", "newton-cg", "liblinear", "sag", "saga"],
            "max_iter": [100, 1000, 2500, 5000],
        },
        "random_forest": {
            "n_estimators": [100, 200, 500],  # number of trees in the forest
            "max_depth": [None, 10, 20, 30],  # maximum depth of each tree
            "min_samples_split": [
                2,
                5,
                10,
            ],  # minimum number of samples required to split an internal node
            "min_samples_leaf": [
                1,
                2,
                4,
            ],  # minimum number of samples required to be at a leaf node
            "max_features": [
                "auto",
                "sqrt",
                "log2",
            ],  # number of features to consider when looking for the best split
            "bootstrap": [
                True,
                False,
            ],  # whether bootstrap samples are used when building trees
        },
        "svc": {
            "C": [0.1, 1, 10, 100, 1000],  # Regularization parameter
            "gamma": [
                1,
                0.1,
                0.01,
                0.001,
                0.0001,
            ],  # Kernel coefficient for 'rbf'
            "kernel": ["rbf"],  # Kernel type to be used
        },
        "knn": {
            "n_neighbors": [3, 5, 7, 9, 11],  # valid integers for neighbors
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": [20, 30, 40],
            "p": [1, 2],  # only 1 or 2 valid for Minkowski metric here
        },
        "gradient_boosting": {
            "n_estimators": [50, 100],  # fewer options for number of trees
            "learning_rate": [0.05, 0.1],  # narrowed learning rates
            "max_depth": [3, 5],  # limited tree depths
            "min_samples_split": [2, 5],  # fewer split thresholds
            "min_samples_leaf": [1, 2],  # fewer leaf sizes
            "subsample": [0.8, 1.0],  # less subsampling options
            "max_features": ["sqrt", None],  # common feature range options
            "n_iter_no_change": [5],  # fixed early stopping iterations
            "tol": [1e-4],  # fixed tolerance
        },
        "naive_bayes": {
            "alpha": [
                0.1,
                0.5,
                1.0,
                5.0,
                10.0,
            ],  # additive smoothing parameter (Laplace/Lidstone smoothing)
            "fit_prior": [
                True,
                False,
            ],  # whether to learn class prior probabilities or not
            "class_prior": [
                None
            ],  # typically left None, but can specify prior class probabilities
        },
        "extra_trees": {
            "n_estimators": [100, 200],  # number of trees
            "criterion": ["gini", "entropy"],  # split quality measure
            "max_depth": [None, 20],  # tree depth limits
            "min_samples_split": [2, 5],  # min samples to split
            "min_samples_leaf": [1, 2],  # min samples at leaf
            "max_features": ["sqrt", "log2"],  # features to consider at splits
            "bootstrap": [False, True],  # whether bootstrap samples are used
        },
        "adaboost": {
            "n_estimators": [50, 100, 150],  # number of weak learners
            "learning_rate": [
                0.01,
                0.1,
                1.0,
            ],  # contribution of each weak learner
            "estimator": [  # base estimators; typically shallow decision trees
                None,
                DecisionTreeClassifier(max_depth=1),
                DecisionTreeClassifier(max_depth=2),
            ],
            "algorithm": ["SAMME", "SAMME.R"],  # boosting algorithm variant
            "random_state": [None, 42],  # random seed for reproducibility
        },
        "mlp": {
            "hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "activation": ["relu", "tanh"],
            "solver": ["adam", "lbfgs"],
            "alpha": [1e-4, 1e-3],
            "learning_rate": ["constant", "adaptive"],
        },
        "LinearDiscriminantAnalysis": {
            "solver": ["svd", "lsqr", "eigen"],
            "shrinkage": [None, "auto"],  # Only used if solver != 'svd'
            "n_components": [
                None,
                1,
                2,
            ],  # Keep None unless you want dimensionality reduction
        },
    }
}
