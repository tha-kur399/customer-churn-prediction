"""
=============================================================================
Script: run_pipeline.py
Description: Master CLI runner to execute the complete Machine Learning workflow:
1. Verify / Generate Benchmark Dataset
2. Ingest, Clean, and Partition Data (80:20 Split)
3. Build & Save Preprocessing Pipeline (Scaling + Encoding)
4. Train & Benchmark Multiple ML Algorithms
5. Generate Evaluation & EDA Visualizations
6. Save Best Model and Preprocessor to `artifacts/`
=============================================================================
"""

import os
import sys
import time

from src.logger import logger
from src.exception import CustomException
from src.pipeline.train_pipeline import TrainPipeline
from generate_dataset import generate_telecom_data
from eda_analysis import run_eda


def main():
    start_time = time.time()
    logger.info("*************************************************************")
    logger.info("Starting Master ML Pipeline Execution")
    logger.info("*************************************************************")
    print("\n" + "="*70)
    print("      CUSTOMER CHURN PREDICTION - MASTER ML PIPELINE")
    print("="*70 + "\n")

    try:
        # Step 1: Ensure Raw Dataset Exists
        raw_data_dir = os.path.join("data", "raw")
        raw_data_file = os.path.join(raw_data_dir, "telecom_churn.csv")
        print("[1/4] Preparing authentic benchmark dataset...")
        os.makedirs(raw_data_dir, exist_ok=True)
        df = generate_telecom_data(num_records=7043)
        df.to_csv(raw_data_file, index=False)
        print(f"      Dataset ready: {raw_data_file} ({df.shape[0]} rows, {df.shape[1]} columns)")

        # Step 2: Run Exploratory Data Analysis Plots
        print("[2/4] Generating Exploratory Data Analysis (EDA) visualizations...")
        run_eda(raw_data_path=raw_data_file)
        print("      Saved EDA plots to 'reports/' and 'webapp/static/images/'")

        # Step 3: Run End-to-End Training Pipeline
        print("[3/4] Running Data Ingestion, Transformation & Multi-Model Training...")
        pipeline = TrainPipeline()
        metrics = pipeline.run_pipeline()

        # Step 4: Display Summary Results
        print("\n" + "-"*70)
        print("                       BENCHMARK RESULTS")
        print("-"*70)
        print(f"{'Model Name':<25} | {'Accuracy':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
        print("-"*70)
        for m_name, m_data in metrics["all_models_comparison"].items():
            print(f"{m_name:<25} | {m_data['test_accuracy']:<10.4f} | {m_data['test_f1']:<10.4f} | {m_data['test_roc_auc']:<10.4f}")
        print("-"*70)
        print(f"\n[WINNER] Best Model: >>> {metrics['best_model_name']} <<<")
        print(f"         Test Accuracy  : {metrics['best_model_metrics']['accuracy'] * 100:.2f}%")
        print(f"         Test F1-Score  : {metrics['best_model_metrics']['f1_score']:.4f}")
        print(f"         Test ROC-AUC   : {metrics['best_model_metrics']['roc_auc']:.4f}")

        elapsed = time.time() - start_time
        print(f"\n[4/4] Pipeline completed successfully in {elapsed:.2f} seconds!")
        print(f"      Saved Artifacts: artifacts/model.joblib & artifacts/preprocessor.joblib")
        print(f"      Log File: logs/churn_system.log\n")

    except Exception as e:
        logger.error(f"Master pipeline execution failed: {str(e)}")
        print(f"\n[ERROR] Pipeline failed! Check logs: {str(e)}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
