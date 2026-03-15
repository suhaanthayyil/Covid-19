"""
Model training pipeline: Logistic Regression, Random Forest, XGBoost, SVM.
5-fold cross-validation on training set; final fit on full train; predict on test.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import xgboost as xgb

# Default configs for the four models
MODEL_CONFIGS = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "n_jobs": -1,
    },
    "Random Forest": {
        "model": RandomForestClassifier(n_estimators=100, random_state=42),
        "n_jobs": -1,
    },
    "XGBoost": {
        "model": xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42, eval_metric="logloss"
        ),
        "n_jobs": -1,
    },
    "SVM": {
        "model": SVC(kernel="rbf", probability=True, random_state=42),
        "n_jobs": -1,
    },
}


def run_cv_and_fit(name: str, model, X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> tuple:
    """
    Run 5-fold cross-validation for accuracy on training set,
    then fit the model on the full training set.
    Returns (fitted_model, mean_cv_accuracy).
    """
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    mean_cv = float(np.mean(cv_scores))
    model.fit(X_train, y_train)
    return model, mean_cv


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
) -> tuple[dict, dict]:
    """
    Train all 4 models: LR, RF, XGBoost, SVM.
    Returns (trained_models_dict, cv_results_dict).
    cv_results_dict: model_name -> mean CV accuracy.
    """
    trained = {}
    cv_results = {}
    for name, config in MODEL_CONFIGS.items():
        print(f"Training {name}...")
        model = config["model"]
        model, mean_cv = run_cv_and_fit(name, model, X_train, y_train, cv=cv)
        trained[name] = model
        cv_results[name] = mean_cv
        print(f"  Mean CV accuracy: {mean_cv:.4f}")
    return trained, cv_results
