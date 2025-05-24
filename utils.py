# utils.py

import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import config


def set_seed(seed=config.RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)


def encode_categorical(df):
    """Encode categorical features using Label Encoding."""
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def split_data(df, test_size=0.2):
    """Split dataframe into train and test sets."""
    X = df.drop(config.TARGET_COL, axis=1)
    y = df[config.TARGET_COL]
    return train_test_split(X, y, test_size=test_size, random_state=config.RANDOM_STATE)


def ensure_dir_exists(path):
    os.makedirs(path, exist_ok=True)