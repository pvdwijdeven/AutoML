import numpy as np
from sklearn.tree import DecisionTreeClassifier

param_grids = {
    "binary_classification": {
        "logistic_regression": {
            "penalty": ["l1", "l2"],  # 2 options
            "C": np.logspace(-2, 2, 5),  # 5 values -> total: 2*5=10
            "solver": ["lbfgs", "saga"],  # 2 options -> total: 10*2=20
            "max_iter": [500, 1000],  # 2 options -> total: 20*2=40
        },
        "random_forest": {
            "n_estimators": [100, 300],  # 2
            "max_depth": [None, 20],  # 2
            "min_samples_split": [2, 10],  # 2
            "min_samples_leaf": [1, 4],  # 2
            "max_features": ["sqrt", "log2"],  # 2
            "bootstrap": [True],  # 1
            # 2*2*2*2*2*1 = 32 combinations
        },
        "svc": {
            "C": [0.1, 1, 10],  # 3
            "gamma": [0.01, 0.001],  # 2
            "kernel": ["rbf"],  # 1
            # 3*2*1=6 combinations
        },
        "knn": {
            "n_neighbors": [3, 7, 11],  # 3
            "weights": ["uniform"],  # 1
            "algorithm": ["auto"],  # 1
            "leaf_size": [30],  # 1
            "p": [1, 2],  # 2
            # 3*1*1*1*2=6 combinations
        },
        "gradient_boosting": {
            "n_estimators": [50, 100],  # fewer options for number of trees
            "learning_rate": [0.05, 0.1],  # narrowed learning rates
            "max_depth": [3, 5],  # limited tree depths
            "min_samples_split": [2, 5],  # fewer split thresholds
            "min_samples_leaf": [8],  # 1
            "subsample": [1.0],  # 1
            "max_features": ["sqrt"],  # 1
            "n_iter_no_change": [9],  # 1
            "tol": [1e-4],  # 1
            # 2*1*2*1*1*1*1*1*1 = 4 combinations
        },
        "naive_bayes": {
            "alpha": [0.1, 1.0],  # 2
            "fit_prior": [True, False],  # 2
            "class_prior": [None],  # 1
            # 2*2*1=4 combinations
        },
        "extra_trees": {
            "n_estimators": [100, 200],  # 2
            "criterion": ["gini"],  # 1
            "max_depth": [None],  # 1
            "min_samples_split": [2, 5],  # 2
            "min_samples_leaf": [1],  # 1
            "max_features": ["sqrt"],  # 1
            "bootstrap": [False],  # 1
            # 2*1*1*2*1*1*1=4 combinations
        },
        "adaboost": {
            "n_estimators": [50, 100],  # 2
            "learning_rate": [0.01, 0.1],  # 2
            "estimator": [None],  # 1 (keep simple here)
            "algorithm": ["SAMME"],  # 1
            "random_state": [None],  # 1
            # 2*2*1*1*1=4 combinations
        },
        "mlp": {
            "hidden_layer_sizes": [(50,), (100,)],  # 2
            "activation": ["relu"],  # 1
            "solver": ["adam"],  # 1
            "alpha": [1e-4],  # 1
            "learning_rate": ["constant"],  # 1
            # 2*1*1*1*1=2 combinations
        },
        "LinearDiscriminantAnalysis": {
            "solver": ["svd", "lsqr"],  # 2
            "shrinkage": [None],  # 1
            "n_components": [None],  # 1
            # 2*1*1=2 combinations
        },
    },
    "regression": {
        "linear_regression": {
            # LinearRegression has few hyperparameters, use fit_intercept and normalize for non-obsolete compatibility
            "fit_intercept": [True, False],  # 2
        },
        "ridge_regression": {
            "alpha": [0.01, 0.1, 1.0, 10.0],  # 4
            "fit_intercept": [True, False],  # 2
            "solver": ["auto", "svd"],  # 2
        },
        "lasso_regression": {
            "alpha": [0.01, 0.1, 1.0, 10.0],  # 4
            "fit_intercept": [True, False],  # 2
            "selection": ["cyclic", "random"],  # 2
        },
        "elasticnet_regression": {
            "alpha": [0.01, 0.1, 1.0, 10.0],  # 4
            "l1_ratio": [0.1, 0.5, 0.9],  # 3
            "fit_intercept": [True, False],  # 2
            "selection": ["cyclic"],  # 1
        },
        "bayesian_ridge_regression": {
            "max_iter": [300, 600],  # 2
            "alpha_1": [1e-6, 1e-4],  # 2
            "lambda_1": [1e-6, 1e-4],  # 2
            "fit_intercept": [True],  # 1
        },
        "random_forest_regressor": {
            "n_estimators": [100, 300],  # 2
            "max_depth": [None, 10],  # 2
            "min_samples_split": [2, 5],  # 2
            "min_samples_leaf": [1, 2],  # 2
            "max_features": ["auto", "sqrt"],  # 2
            "bootstrap": [True],  # 1
        },
        "gradient_boosting_regressor": {
            "n_estimators": [50, 100],  # 2
            "learning_rate": [0.05, 0.1],  # 2
            "max_depth": [3, 5],  # 2
            "min_samples_split": [2],  # 1
            "min_samples_leaf": [2],  # 1
            "subsample": [1.0],  # 1
            "max_features": ["sqrt"],  # 1
        },
        "svr": {
            "C": [0.1, 1, 10],  # 3
            "kernel": ["rbf", "linear"],  # 2
            "gamma": ["scale"],  # 1
            "epsilon": [0.01, 0.1],  # 2
        },
        "knn_regressor": {
            "n_neighbors": [3, 5, 7],  # 3
            "weights": ["uniform", "distance"],  # 2
            "algorithm": ["auto"],  # 1
            "leaf_size": [30],  # 1
            "p": [1, 2],  # 2
        },
        "mlp_regressor": {
            "hidden_layer_sizes": [(50,), (100,)],  # 2
            "activation": ["relu"],  # 1
            "solver": ["adam"],  # 1
            "alpha": [1e-4],  # 1
            "learning_rate": ["constant"],  # 1
            "max_iter": [300, 500],  # 2
        },
        "xgboost_regressor": {
            "n_estimators": [50, 100],  # 2
            "max_depth": [3, 6],  # 2
            "learning_rate": [0.05, 0.1],  # 2
            "subsample": [0.8, 1.0],  # 2
            "colsample_bytree": [0.8, 1.0],  # 2
            "reg_alpha": [0, 0.1],  # 2
            "reg_lambda": [0.5, 1.0],  # 2
        },
    },
}


param_grids_detailed = {
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
    },
    "regression": {
        "linear_regression": {
            "fit_intercept": [True, False],
            "copy_X": [True, False],
            "positive": [True, False],
        },
        "ridge_regression": {
            "alpha": np.logspace(-3, 2, 6),  # [0.001, ..., 100]
            "fit_intercept": [True, False],
            "solver": [
                "auto",
                "svd",
                "cholesky",
                "lsqr",
                "sparse_cg",
                "sag",
                "saga",
            ],
        },
        "lasso_regression": {
            "alpha": np.logspace(-3, 1, 5),  # [0.001, ..., 10]
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "selection": ["cyclic", "random"],
            "max_iter": [500, 1000, 2500],
        },
        "elasticnet_regression": {
            "alpha": np.logspace(-3, 1, 5),
            "l1_ratio": [0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "selection": ["cyclic", "random"],
            "max_iter": [500, 1000, 2500],
        },
        "bayesian_ridge_regression": {
            "max_iter": [300, 400, 500, 600, 1000],
            "alpha_1": [1e-6, 1e-5, 1e-4],
            "alpha_2": [1e-6, 1e-5, 1e-4],
            "lambda_1": [1e-6, 1e-5, 1e-4],
            "lambda_2": [1e-6, 1e-5, 1e-4],
            "fit_intercept": [True, False],
        },
        "random_forest_regressor": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["auto", "sqrt", "log2"],
            "bootstrap": [True, False],
        },
        "gradient_boosting_regressor": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "subsample": [0.8, 1.0],
            "max_features": ["auto", "sqrt", None],
            "n_iter_no_change": [5, 10],
            "tol": [1e-4, 1e-3],
        },
        "svr": {
            "C": np.logspace(-2, 2, 5),
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "gamma": ["scale", "auto"],
            "epsilon": [0.01, 0.1, 0.5, 1.0],
            "degree": [2, 3, 4],
        },
        "knn_regressor": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": [20, 30, 40],
            "p": [1, 2],
        },
        "mlp_regressor": {
            "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 100)],
            "activation": ["relu", "tanh"],
            "solver": ["adam", "lbfgs"],
            "alpha": [1e-5, 1e-4, 1e-3],
            "learning_rate": ["constant", "adaptive"],
            "max_iter": [500, 1000, 2500],
        },
        "xgboost_regressor": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "reg_alpha": [0, 0.1, 0.5, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0],
        },
    },
}
