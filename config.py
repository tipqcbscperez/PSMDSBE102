# config.py

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# Dataset
RAW_DATA_PATH = os.path.join(DATA_DIR, "Telco-Customer-Churn.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_data.csv")

# Model
MODEL_NAME = "random_forest_churn"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.pkl")

# MLflow
MLFLOW_TRACKING_URI = "http://localhost:5000"  # Adjust if hosted elsewhere
MLFLOW_EXPERIMENT_NAME = "Telco-Churn-Experiment"

# Ray Tune
RAY_NUM_SAMPLES = 20
RAY_CPUS_PER_TRIAL = 2
RAY_GPUS_PER_TRIAL = 0

# Random Forest default params
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "random_state": 42
}

# General
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "Churn"