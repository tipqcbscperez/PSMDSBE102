# data.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import config


def load_data(path=config.RAW_DATA_PATH):
    """Load the raw dataset from CSV."""
    return pd.read_csv(path)


def preprocess_data(df):
    """Preprocess the Telco Customer Churn dataset."""
    df = df.copy()

    # Drop customerID
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges to numeric, coercing errors
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Convert target to binary
    df[config.TARGET_COL] = df[config.TARGET_COL].map({"Yes": 1, "No": 0})

    # Label encode categorical features
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def split_data(df):
    """Split data into train and test sets."""
    X = df.drop(config.TARGET_COL, axis=1)
    y = df[config.TARGET_COL]
    return train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)


def load_and_prepare():
    """Convenience function to load, preprocess and split."""
    df = load_data()
    df = preprocess_data(df)
    return split_data(df)
