# train.py

import argparse
import joblib
import mlflow
import mlflow.sklearn

from data import load_and_prepare
from models import RandomForestClassifier, get_model
from evaluate import evaluate_model, print_evaluation
import config


def parse_args():
    parser = argparse.ArgumentParser(description="Train RandomForest model for Telco Churn")
    parser.add_argument("--n-estimators", type=int, default=config.RF_PARAMS["n_estimators"],
                        help="Number of trees in the forest")
    parser.add_argument("--max-depth", type=int, default=config.RF_PARAMS["max_depth"],
                        help="Maximum depth of each tree")
    parser.add_argument("--min-samples-split", type=int, default=2,
                        help="Minimum number of samples required to split an internal node")
    parser.add_argument("--max-features", type=str, default="auto",
                        help="Number of features to consider at each split")
    parser.add_argument("--save-local", action="store_true",
                        help="Whether to save the trained model locally as a pickle")
    return parser.parse_args()


def main():
    args = parse_args()

    # Update RF parameters
    rf_params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "max_features": args.max_features,
        "random_state": config.RANDOM_STATE
    }

    # Start MLflow run
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        # Load data
        X_train, X_test, y_train, y_test = load_and_prepare()

        # Build and train model
        model = get_model() if rf_params == config.RF_PARAMS else RandomForestClassifier(**rf_params)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        print_evaluation(metrics)

        # Log metrics and model
        mlflow.log_params(rf_params)
        mlflow.sklearn.log_model(model, artifact_path=config.MODEL_NAME)

        # Optionally save locally
        if args.save_local:
            joblib.dump(model, config.MODEL_SAVE_PATH)
            print(f"Model saved locally at {config.MODEL_SAVE_PATH}")

        print("Training run complete.")


if __name__ == "__main__":
    main()