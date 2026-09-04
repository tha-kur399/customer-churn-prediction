"""
=============================================================================
Module: src/utils.py
Description: Common utility functions for saving/loading serialized artifacts,
evaluating ML algorithms, and computing performance metrics.
=============================================================================
"""

import os
import sys
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.exception import CustomException
from src.logger import logger


def save_object(file_path: str, obj: object) -> None:
    """
    Saves a Python object (e.g. Scikit-learn model, Preprocessor) to disk.

    Args:
        file_path (str): Destination path.
        obj (object): Python object to serialize.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info(f"Successfully serialized object at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str) -> object:
    """
    Loads a serialized Python object from disk.

    Args:
        file_path (str): Source path of the .joblib file.

    Returns:
        object: Deserialized Python object.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The specified artifact file was not found: {file_path}")
        obj = joblib.load(file_path)
        logger.info(f"Successfully loaded object from: {file_path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    models: dict) -> dict:
    """
    Trains multiple models, evaluates their performance across 5 key metrics,
    and returns a detailed comparison dictionary.

    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training target
        X_test (np.ndarray): Testing features
        y_test (np.ndarray): Testing target
        models (dict): Mapping of model name -> model instance

    Returns:
        dict: Detailed results and metrics for each algorithm.
    """
    try:
        report = {}
        for model_name, model in models.items():
            logger.info(f"Initiating training for algorithm: [{model_name}]")
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Probabilities for ROC-AUC
            if hasattr(model, "predict_proba"):
                y_train_proba = model.predict_proba(X_train)[:, 1]
                y_test_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_train_proba = y_train_pred
                y_test_proba = y_test_pred

            # Performance Metrics on Test Set
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred, zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
            try:
                test_roc_auc = roc_auc_score(y_test, y_test_proba)
            except Exception:
                test_roc_auc = 0.5

            # Training accuracy to detect overfitting
            train_accuracy = accuracy_score(y_train, y_train_pred)

            report[model_name] = {
                "train_accuracy": round(float(train_accuracy), 4),
                "test_accuracy": round(float(test_accuracy), 4),
                "test_precision": round(float(test_precision), 4),
                "test_recall": round(float(test_recall), 4),
                "test_f1": round(float(test_f1), 4),
                "test_roc_auc": round(float(test_roc_auc), 4),
                "model_instance": model
            }

            logger.info(f"[{model_name}] Results -> Acc: {test_accuracy:.4f} | F1: {test_f1:.4f} | ROC-AUC: {test_roc_auc:.4f}")

        return report
    except Exception as e:
        raise CustomException(e, sys)
