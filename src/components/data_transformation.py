"""
=============================================================================
Module: src/components/data_transformation.py
Description: Data transformation, encoding, feature scaling, and preprocessing
pipeline construction using Scikit-Learn ColumnTransformer.
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    """
    Configuration dataclass specifying the serialized preprocessor artifact path.
    """
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.joblib")


class DataTransformation:
    """
    Handles data cleaning, missing value imputation, one-hot encoding,
    and feature scaling for both training and inference.
    """
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Constructs and returns the Scikit-Learn ColumnTransformer pipeline
        with separate sub-pipelines for numerical and categorical attributes.

        Returns:
            ColumnTransformer: Configured preprocessor pipeline object.
        """
        try:
            # Define numerical features
            numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

            # Define categorical features
            categorical_columns = [
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod"
            ]

            logger.info(f"Numerical columns for scaling: {numerical_columns}")
            logger.info(f"Categorical columns for encoding: {categorical_columns}")

            # Numerical Pipeline: Impute missing with median -> Standardize
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical Pipeline: Impute missing with mode -> One-Hot Encode
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                ]
            )

            # Combine numerical and categorical transformations
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ],
                remainder="drop"
            )

            logger.info("ColumnTransformer pipeline constructed successfully.")
            return preprocessor

        except Exception as e:
            logger.error("Error creating Data Transformer object.")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str) -> tuple[np.ndarray, np.ndarray, str]:
        """
        Reads train and test CSVs, applies preprocessing pipelines, fits on train set,
        transforms test set, and saves the fitted preprocessor artifact.

        Args:
            train_path (str): Path to train.csv
            test_path (str): Path to test.csv

        Returns:
            tuple: (train_arr, test_arr, preprocessor_obj_file_path)
        """
        logger.info(">>> Data Transformation component initialized.")
        try:
            # 1. Read train and test data
            logger.info(f"Loading train dataset: {train_path}")
            train_df = pd.read_csv(train_path)

            logger.info(f"Loading test dataset: {test_path}")
            test_df = pd.read_csv(test_path)

            # 2. Convert TotalCharges to numeric (handles potential whitespace/null strings)
            train_df["TotalCharges"] = pd.to_numeric(train_df["TotalCharges"], errors="coerce")
            test_df["TotalCharges"] = pd.to_numeric(test_df["TotalCharges"], errors="coerce")

            # 3. Instantiate Preprocessor object
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Churn"
            drop_columns = ["customerID", target_column_name]

            # 4. Separate Input Features (X) and Target (y)
            input_feature_train_df = train_df.drop(columns=[col for col in drop_columns if col in train_df.columns])
            target_feature_train_df = train_df[target_column_name].map({"Yes": 1, "No": 0})

            input_feature_test_df = test_df.drop(columns=[col for col in drop_columns if col in test_df.columns])
            target_feature_test_df = test_df[target_column_name].map({"Yes": 1, "No": 0})

            logger.info("Applying transformation pipeline to Train and Test sets.")

            # 5. Fit & Transform on Train set; Transform on Test set (Avoid Data Leakage)
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # 6. Concatenate features with target array
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logger.info(f"Train Array shape: {train_arr.shape} | Test Array shape: {test_arr.shape}")

            # 7. Persist preprocessor object to disk
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logger.info(">>> Data Transformation completed successfully.")
            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logger.error("Error during Data Transformation phase.")
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    ingestion = DataIngestion()
    tr_path, ts_path = ingestion.initiate_data_ingestion()
    transformer = DataTransformation()
    tr_arr, ts_arr, pre_path = transformer.initiate_data_transformation(tr_path, ts_path)
    print(f"[Transformation Complete] Train shape: {tr_arr.shape}, Test shape: {ts_arr.shape}")
