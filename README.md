# Telco Customer Churn Prediction

This project uses the Telco Customer Churn dataset (from Kaggle) to predict churn using a Random Forest classifier, with support for hyperparameter tuning (Ray Tune), experiment tracking (MLflow), and real-time serving (FastAPI).

---

## 🔧 Project Structure

```
.
├── config.py            # Configuration settings
├── data.py              # Data loading and preprocessing
├── evaluate.py          # Evaluation metrics
├── main.py              # Entry point for basic training
├── models.py            # Model definition and Ray-compatible training
├── predict.py           # CLI tool to run batch predictions
├── serve.py             # FastAPI server for real-time inference
├── train.py             # MLflow-enabled training script
├── tune.py              # Ray Tune hyperparameter tuning
├── utils.py             # Utility functions
└── README.md            # Project documentation
```

---

## 📦 Requirements

```bash
pip install -r requirements.txt
```

Suggested packages:

* `scikit-learn`
* `pandas`, `numpy`
* `mlflow`
* `ray[tune]`
* `fastapi`, `uvicorn`

---

## 🚀 Training the Model

```bash
python train.py --n-estimators 100 --max-depth 10 --save-local
```

Logs metrics and model artifacts to MLflow.

---

## 🔍 Hyperparameter Tuning with Ray

```bash
python tune.py
```

Uses Ray Tune with ASHA scheduler to optimize model performance.

---

## 🔮 Running Predictions

```bash
python predict.py \
    --input-csv ./data/new_customers.csv \
    --output-csv ./output/predictions.csv
```

---

## 🌐 Serving the Model

```bash
uvicorn serve:app --reload
```

Once running, test via curl or Postman:

```bash
curl -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{
          "gender": "Female",
          "SeniorCitizen": 0,
          "Partner": "Yes",
          "Dependents": "No",
          "tenure": 5,
          ...
        }'
```

---

## 📁 MLflow Tracking

Set your MLflow URI in `config.py`. Launch the UI with:

```bash
mlflow ui
```

Then visit `http://localhost:5000` to explore runs and metrics.

---

## 📌 Notes

* Assumes the dataset follows the structure of the Kaggle Telco dataset.
* Requires `data/preprocess_data()` to match the training pipeline.
* You can extend model support in `models.py`.

---

## ✅ TODOs

* Make it work.

---

## 📜 License

MIT License.
