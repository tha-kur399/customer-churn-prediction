"""
=============================================================================
Module: src/pipeline/train_pipeline.py
Description: End-to-End Training Pipeline. Orchestrates Data Ingestion,
Data Transformation, and Model Training & Benchmarking.
=============================================================================
"""

import sys
from src.logger import logger
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    """
    Orchestration pipeline class to run complete model training workflow.
    """
    def __init__(self):
        logger.info("Initializing TrainPipeline instance.")

    def run_pipeline(self) -> dict:
        """
        Executes ingestion -> transformation -> model training sequentially.

        Returns:
            dict: Summary metrics of the trained models.
        """
        try:
            logger.info("=====================================================")
            logger.info("Starting End-to-End Training Pipeline Execution")
            logger.info("=====================================================")

            # Step 1: Data Ingestion
            ingestion = DataIngestion()
            train_data_path, test_data_path = ingestion.initiate_data_ingestion()

            # Step 2: Data Transformation
            transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
                train_path=train_data_path,
                test_path=test_data_path
            )

            # Step 3: Model Training & Evaluation
            trainer = ModelTrainer()
            metrics_summary = trainer.initiate_model_trainer(
                train_array=train_arr,
                test_array=test_arr,
                preprocessor_path=preprocessor_path
            )

            logger.info("=====================================================")
            logger.info(f"Training Pipeline Completed! Best Model: {metrics_summary['best_model_name']}")
            logger.info("=====================================================")

            return metrics_summary

        except Exception as e:
            logger.error("Training Pipeline failed during execution.")
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    metrics = pipeline.run_pipeline()
    print("\n[SUCCESS] Pipeline Executed Successfully!")
    print(f"Best Model: {metrics['best_model_name']}")
    print(f"Metrics: {metrics['best_model_metrics']}")
