# evaluate.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow


def evaluate_model(y_true, y_pred, log_to_mlflow=True):
    """Evaluate model performance and optionally log to MLflow."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    if log_to_mlflow:
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm
    }


def print_evaluation(metrics_dict):
    """Prints evaluation metrics nicely."""
    print("Model Evaluation Metrics:")
    for k, v in metrics_dict.items():
        if k != "confusion_matrix":
            print(f"  {k.capitalize()}: {v:.4f}")
        else:
            print(f"  Confusion Matrix:\n{v}")