"""
=============================================================================
Module: src/components/model_trainer.py
Description: Trains and evaluates multiple Machine Learning algorithms,
selects the optimal model based on ROC-AUC / F1-Score, generates evaluation
visualizations (Confusion Matrix, ROC Curve, Feature Importance), and serializes
the best model artifact.
=============================================================================
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    """
    Configuration paths for model artifacts, metrics, and visualization plots.
    """
    trained_model_file_path: str = os.path.join("artifacts", "model.joblib")
    metrics_file_path: str = os.path.join("artifacts", "metrics.json")
    reports_dir: str = os.path.join("reports")
    static_images_dir: str = os.path.join("webapp", "static", "images")


class ModelTrainer:
    """
    Handles multi-model training, hyperparameter configuration, metric calculation,
    visual artifact generation, and best model selection.
    """
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        os.makedirs(self.model_trainer_config.reports_dir, exist_ok=True)
        os.makedirs(self.model_trainer_config.static_images_dir, exist_ok=True)

    def generate_evaluation_plots(self, model, X_test, y_test, best_model_name: str, feature_names: list = None):
        """
        Generates and saves Confusion Matrix, ROC-AUC Curve, and Feature Importance plots.
        """
        try:
            logger.info("Generating model evaluation visualization plots.")
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

            # Set styling
            plt.style.use("dark_background")

            # 1. Confusion Matrix Plot
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5), facecolor="#111827")
            ax = sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Retained (0)", "Churned (1)"],
                yticklabels=["Retained (0)", "Churned (1)"],
                cbar=False
            )
            ax.set_facecolor("#1f2937")
            plt.title(f"Confusion Matrix - {best_model_name}", fontsize=14, color="#f3f4f6", pad=15)
            plt.xlabel("Predicted Class", fontsize=12, color="#9ca3af")
            plt.ylabel("Actual Class", fontsize=12, color="#9ca3af")
            plt.tight_layout()

            cm_path_rep = os.path.join(self.model_trainer_config.reports_dir, "confusion_matrix.png")
            cm_path_static = os.path.join(self.model_trainer_config.static_images_dir, "confusion_matrix.png")
            plt.savefig(cm_path_rep, dpi=200)
            plt.savefig(cm_path_static, dpi=200)
            plt.close()

            # 2. ROC-AUC Curve Plot
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = roc_auc_score(y_test, y_proba)

            plt.figure(figsize=(6, 5), facecolor="#111827")
            plt.plot(fpr, tpr, color="#38bdf8", lw=2.5, label=f"ROC curve (AUC = {auc_score:.3f})")
            plt.plot([0, 1], [0, 1], color="#6b7280", linestyle="--", lw=1.5)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.title(f"ROC Curve - {best_model_name}", fontsize=14, color="#f3f4f6", pad=15)
            plt.xlabel("False Positive Rate", fontsize=12, color="#9ca3af")
            plt.ylabel("True Positive Rate", fontsize=12, color="#9ca3af")
            plt.legend(loc="lower right", facecolor="#1f2937", edgecolor="#374151", labelcolor="#f3f4f6")
            plt.grid(alpha=0.2, color="#4b5563")
            plt.tight_layout()

            roc_path_rep = os.path.join(self.model_trainer_config.reports_dir, "roc_curve.png")
            roc_path_static = os.path.join(self.model_trainer_config.static_images_dir, "roc_curve.png")
            plt.savefig(roc_path_rep, dpi=200)
            plt.savefig(roc_path_static, dpi=200)
            plt.close()

            # 3. Feature Importance Plot (if applicable)
            if hasattr(model, "feature_importances_") and feature_names is not None:
                importances = model.feature_importances_
                top_indices = np.argsort(importances)[-15:]  # Top 15 features

                plt.figure(figsize=(8, 6), facecolor="#111827")
                plt.barh(
                    [feature_names[i] for i in top_indices],
                    importances[top_indices],
                    color="#818cf8",
                    edgecolor="#6366f1"
                )
                plt.title("Top 15 Feature Importances", fontsize=14, color="#f3f4f6", pad=15)
                plt.xlabel("Importance Score", fontsize=12, color="#9ca3af")
                plt.tight_layout()

                fi_path_rep = os.path.join(self.model_trainer_config.reports_dir, "feature_importance.png")
                fi_path_static = os.path.join(self.model_trainer_config.static_images_dir, "feature_importance.png")
                plt.savefig(fi_path_rep, dpi=200)
                plt.savefig(fi_path_static, dpi=200)
                plt.close()

            logger.info("Evaluation plots generated and saved successfully.")
        except Exception as e:
            logger.warning(f"Plot generation non-fatal warning: {str(e)}")

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray, preprocessor_path: str = None) -> dict:
        """
        Trains candidate models, benchmarks performance, logs results,
        and saves the top performing model artifact.

        Args:
            train_array (np.ndarray): Transformed training data (X + y)
            test_array (np.ndarray): Transformed test data (X + y)
            preprocessor_path (str): Optional path to preprocessor for feature name extraction

        Returns:
            dict: Summary metrics of the best model and all candidate models.
        """
        logger.info(">>> Model Trainer component initialized.")
        try:
            # 1. Separate X and y
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # 2. Define Model Candidates with robust parameters
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
                "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced"),
                "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, class_weight="balanced"),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42),
                "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
            }

            # 3. Evaluate all candidate models
            logger.info("Benchmarking candidate classification algorithms.")
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            # 4. Determine best model based on combined F1-Score and ROC-AUC
            best_model_name = None
            best_composite_score = -1.0

            for name, results in model_report.items():
                composite_score = (results["test_f1"] * 0.5) + (results["test_roc_auc"] * 0.5)
                logger.info(f"Model: {name} -> Composite Score (F1+AUC): {composite_score:.4f}")
                if composite_score > best_composite_score:
                    best_composite_score = composite_score
                    best_model_name = name

            best_model_data = model_report[best_model_name]
            best_model_instance = best_model_data["model_instance"]

            logger.info(f"[BEST MODEL] Best Performing Model Identified: [{best_model_name}] with ROC-AUC: {best_model_data['test_roc_auc']} and F1: {best_model_data['test_f1']}")

            # 5. Extract Feature Names from preprocessor if available
            feature_names = None
            if preprocessor_path and os.path.exists(preprocessor_path):
                try:
                    import joblib
                    preprocessor = joblib.load(preprocessor_path)
                    if hasattr(preprocessor, "get_feature_names_out"):
                        feature_names = [f.split("__")[-1] for f in preprocessor.get_feature_names_out()]
                except Exception as e:
                    logger.warning(f"Could not extract feature names: {e}")

            # 6. Generate evaluation plots
            self.generate_evaluation_plots(
                model=best_model_instance,
                X_test=X_test,
                y_test=y_test,
                best_model_name=best_model_name,
                feature_names=feature_names
            )

            # 7. Save best model artifact
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model_instance
            )

            # 8. Clean report dictionary for JSON storage
            json_metrics = {
                "best_model_name": best_model_name,
                "best_model_metrics": {
                    "accuracy": best_model_data["test_accuracy"],
                    "precision": best_model_data["test_precision"],
                    "recall": best_model_data["test_recall"],
                    "f1_score": best_model_data["test_f1"],
                    "roc_auc": best_model_data["test_roc_auc"]
                },
                "all_models_comparison": {}
            }

            for m_name, m_data in model_report.items():
                json_metrics["all_models_comparison"][m_name] = {
                    "train_accuracy": m_data["train_accuracy"],
                    "test_accuracy": m_data["test_accuracy"],
                    "test_precision": m_data["test_precision"],
                    "test_recall": m_data["test_recall"],
                    "test_f1": m_data["test_f1"],
                    "test_roc_auc": m_data["test_roc_auc"]
                }

            with open(self.model_trainer_config.metrics_file_path, "w", encoding="utf-8") as f:
                json.dump(json_metrics, f, indent=4)

            logger.info(f"Metrics saved to: {self.model_trainer_config.metrics_file_path}")
            logger.info(">>> Model Trainer completed successfully.")

            return json_metrics

        except Exception as e:
            logger.error("Error encountered in Model Trainer component.")
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    ingestion = DataIngestion()
    tr_path, ts_path = ingestion.initiate_data_ingestion()

    transformer = DataTransformation()
    tr_arr, ts_arr, pre_path = transformer.initiate_data_transformation(tr_path, ts_path)

    trainer = ModelTrainer()
    metrics = trainer.initiate_model_trainer(tr_arr, ts_arr, pre_path)
    print(f"[Model Training Complete] Best Model: {metrics['best_model_name']}")
