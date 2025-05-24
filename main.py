# main.py

import mlflow
import mlflow.sklearn
from data import load_and_prepare
from models import get_model
from evaluate import evaluate_model, print_evaluation
import config


def main():
    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        print("Loading and preparing data...")
        X_train, X_test, y_train, y_test = load_and_prepare()

        print("Training model...")
        model = get_model()
        model.fit(X_train, y_train)

        print("Evaluating model...")
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        print_evaluation(metrics)

        print("Logging model to MLflow...")
        mlflow.sklearn.log_model(model, config.MODEL_NAME)

        print("Run complete.")


if __name__ == "__main__":
    main()