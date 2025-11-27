"""
Training and inference utilities for gradient boosting models.
"""

from __future__ import annotations

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def train_classifier(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, shuffle=False)
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True)
    return model, report


def save_model(model, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)
