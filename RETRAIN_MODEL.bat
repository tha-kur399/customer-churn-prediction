@echo off
title Retrain Customer Churn Machine Learning Pipeline
color 0A
cls
echo ===============================================================================
echo            RE-TRAINING MACHINE LEARNING BENCHMARK PIPELINE
echo ===============================================================================
echo.
echo [*] Running Data Ingestion, Transformation, Multi-Model Training and EDA...
echo.
python run_pipeline.py
echo.
echo ===============================================================================
echo [SUCCESS] Training pipeline completed! All models and visual plots updated.
echo ===============================================================================
echo.
pause
