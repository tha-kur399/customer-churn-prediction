"""
=============================================================================
Module: src/components/data_ingestion.py
Description: Handles reading raw dataset, performing basic data validation,
splitting into train and test sets, and persisting processed files.
=============================================================================
"""

import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    """
    Configuration dataclass containing file paths for data ingestion artifacts.
    """
    raw_data_path: str = os.path.join("data", "raw", "telecom_churn.csv")
    train_data_path: str = os.path.join("data", "processed", "train.csv")
    test_data_path: str = os.path.join("data", "processed", "test.csv")


class DataIngestion:
    """
    Class responsible for loading raw data, validating integrity,
    and partitioning into train/test sets.
    """
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> tuple[str, str]:
        """
        Executes data ingestion process:
        1. Reads raw CSV file
        2. Validates structure and target column
        3. Creates train-test split (80-20 stratified)
        4. Saves train and test CSVs

        Returns:
            tuple[str, str]: (train_data_path, test_data_path)
        """
        logger.info(">>> Data Ingestion component initialized.")
        try:
            # 1. Read dataset
            logger.info(f"Reading raw dataset from: {self.ingestion_config.raw_data_path}")
            if not os.path.exists(self.ingestion_config.raw_data_path):
                raise FileNotFoundError(f"Raw data file not found at: {self.ingestion_config.raw_data_path}")

            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logger.info(f"Dataset loaded successfully with shape: {df.shape}")

            # 2. Ensure target column exists
            if "Churn" not in df.columns:
                raise ValueError("Target column 'Churn' not found in dataset!")

            # 3. Create destination directory for processed datasets
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # 4. Stratified Train-Test Split (80% Train, 20% Test)
            logger.info("Performing 80:20 stratified train-test split.")
            train_set, test_set = train_test_split(
                df,
                test_size=0.20,
                random_state=42,
                stratify=df["Churn"]
            )

            # 5. Save partitioned datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logger.info(f"Train dataset saved to: {self.ingestion_config.train_data_path} (Shape: {train_set.shape})")
            logger.info(f"Test dataset saved to: {self.ingestion_config.test_data_path} (Shape: {test_set.shape})")
            logger.info(">>> Data Ingestion completed successfully.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logger.error("Error encountered during Data Ingestion phase.")
            raise CustomException(e, sys)


if __name__ == "__main__":
    ingestion = DataIngestion()
    train_data, test_data = ingestion.initiate_data_ingestion()
    print(f"[Ingestion Complete] Train path: {train_data}, Test path: {test_data}")
