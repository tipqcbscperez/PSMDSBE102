# predict.py

import argparse
import pandas as pd
import joblib
import mlflow.sklearn

from data import preprocess_data
import config


def load_model():
    """Load the trained model from MLflow registry or local file system."""
    # Try MLflow model registry (registered model)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    try:
        model = mlflow.sklearn.load_model(f"models:/{config.MODEL_NAME}/latest")
        print("Loaded model from MLflow registry.")
    except Exception:
        # Fallback to loading from local pickle
        model = joblib.load(config.MODEL_SAVE_PATH)
        print(f"Loaded model from local path: {config.MODEL_SAVE_PATH}")
    return model


def predict_from_df(df, model):
    """Generate predictions and probabilities (if supported)."""
    preds = model.predict(df)
    # Check if model supports predict_proba
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)[:, 1]
    else:
        probs = None
    return preds, probs


def main():
    parser = argparse.ArgumentParser(
        description="Predict churn using a trained RandomForest model"
    )
    parser.add_argument(
        "--input-csv", type=str, required=True,
        help="Path to input CSV file containing raw customer data"
    )
    parser.add_argument(
        "--output-csv", type=str, required=True,
        help="Path where prediction results will be saved as CSV"
    )
    args = parser.parse_args()

    # Load and preprocess input data
    raw_df = pd.read_csv(args.input_csv)
    df = preprocess_data(raw_df)
    # Drop target column if present
    if config.TARGET_COL in df.columns:
        df = df.drop(config.TARGET_COL, axis=1)

    # Load model and perform predictions
    model = load_model()
    preds, probs = predict_from_df(df, model)

    # Prepare output
    output = raw_df.copy()
    output["prediction"] = preds
    if probs is not None:
        output["churn_probability"] = probs

    # Save results
    output.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")


if __name__ == "__main__":
    main()