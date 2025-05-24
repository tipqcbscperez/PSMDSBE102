# tune.py

import os
import mlflow
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from data import load_and_prepare
from models import train_random_forest
import config


def tune_model():
    # Set MLflow tracking
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    # Load and split data
    X_train, X_val, y_train, y_val = load_and_prepare()

    # Define search space for hyperparameters
    search_space = {
        "n_estimators": tune.choice([50, 100, 200]),
        "max_depth": tune.choice([5, 10, 20, None]),
        "min_samples_split": tune.randint(2, 10),
        "max_features": tune.choice(["auto", "sqrt", "log2"])  
    }

    # Scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="val_accuracy",
        mode="max",
        max_t=10,
        grace_period=1,
        reduction_factor=2
    )

    # Run Ray Tune
    analysis = tune.run(
        tune.with_parameters(
            train_random_forest,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        ),
        resources_per_trial={"cpu": config.RAY_CPUS_PER_TRIAL, "gpu": config.RAY_GPUS_PER_TRIAL},
        metric="val_accuracy",
        mode="max",
        config=search_space,
        num_samples=config.RAY_NUM_SAMPLES,
        scheduler=scheduler,
        local_dir=os.path.join(config.ARTIFACTS_DIR, "ray_results"),
        name="rf_hyperparam_tuning"
    )

    # Retrieve best hyperparameters
    best_config = analysis.get_best_config(metric="val_accuracy", mode="max")
    print("Best hyperparameters found:", best_config)

    # Optionally: retrain the best model and log
    print("Retraining best model on full training set...")
    best_model = train_random_forest(best_config, X_train, y_train, X_val, y_val)
    return best_config


if __name__ == "__main__":
    best = tune_model()
    print(f"Tuning complete. Best config: {best}")