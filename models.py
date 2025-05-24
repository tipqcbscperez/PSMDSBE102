# models.py

import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from ray import train

import config


def get_model():
    """Returns a RandomForest model using default config."""
    return RandomForestClassifier(**config.RF_PARAMS)


def train_random_forest(config_dict, X_train, y_train, X_val, y_val):
    """Train and evaluate a RandomForest model, designed for Ray Tune."""
    with mlflow.start_run(nested=True):
        model = RandomForestClassifier(
            n_estimators=config_dict["n_estimators"],
            max_depth=config_dict["max_depth"],
            min_samples_split=config_dict["min_samples_split"],
            max_features=config_dict["max_features"],
            random_state=config.RANDOM_STATE
        )

        # Cross-validation (optional, not used for reporting below)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        mean_cv_score = np.mean(cv_scores)

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)

        # Log metrics and model to MLflow
        mlflow.log_metric("mean_cv_accuracy", mean_cv_score)
        mlflow.log_metric("val_accuracy", acc)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Report metric to Ray Tune
        train.report({"val_accuracy": acc})