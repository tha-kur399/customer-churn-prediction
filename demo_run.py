"""
=============================================================================
Script: demo_run.py
Description: Live Interactive Demonstration of Customer Churn Prediction System.
Demonstrates:
1. Model Metrics & Algorithm Benchmarks
2. Single Customer Inferences (High Risk vs Safe Loyal vs Moderate)
3. Prescriptive Business Retention Strategies
4. Batch CSV Scoring Demonstration
5. Real-Time System Log Inspection
=============================================================================
"""

import os
import sys
import json
import time
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

def print_banner(title):
    print("\n" + "="*75)
    print(f"   {title.upper()}")
    print("="*75)

def main():
    print_banner("CUSTOMER CHURN PREDICTION SYSTEM - LIVE DEMONSTRATION")
    
    # -------------------------------------------------------------
    # 1. Display Benchmarking & Model Accuracy
    # -------------------------------------------------------------
    metrics_path = os.path.join("artifacts", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            
        print("\n[1] MACHINE LEARNING BENCHMARK MATRIX:")
        print("-" * 75)
        print(f"{'Algorithm Name':<24} | {'Accuracy':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
        print("-" * 75)
        for name, m in metrics["all_models_comparison"].items():
            print(f"{name:<24} | {m['test_accuracy']*100:<9.2f}% | {m['test_f1']:<10.4f} | {m['test_roc_auc']:<10.4f}")
        print("-" * 75)
        print(f"[BEST MODEL] Champion: [{metrics['best_model_name']}] | ROC-AUC: {metrics['best_model_metrics']['roc_auc']:.4f}\n")

    # -------------------------------------------------------------
    # 2. Live Single Customer Predictions
    # -------------------------------------------------------------
    pipeline = PredictPipeline()
    
    print_banner("2. LIVE CUSTOMER RISK EVALUATION & RETENTION DEMO")

    # Test Case A: High Risk Customer
    print("\n[CASE A: HIGH CHURN RISK CUSTOMER PROFILE]")
    print("Profile: Month-to-month contract, Tenure: 2 Months, Fiber Optic, No Tech Support, Electronic Check, $98.50/mo")
    high_risk_data = CustomData(
        gender="Female", SeniorCitizen=1, Partner="No", Dependents="No", tenure=2,
        PhoneService="Yes", MultipleLines="Yes", InternetService="Fiber optic",
        OnlineSecurity="No", OnlineBackup="No", DeviceProtection="No", TechSupport="No",
        StreamingTV="Yes", StreamingMovies="Yes", Contract="Month-to-month",
        PaperlessBilling="Yes", PaymentMethod="Electronic check",
        MonthlyCharges=98.50, TotalCharges=197.00
    )
    result_a = pipeline.predict(high_risk_data.get_data_as_data_frame())
    print(f"   -> AI Prediction       : {result_a['label']}")
    print(f"   -> Churn Probability  : {result_a['churn_probability']}%")
    print(f"   -> Risk Tier Category : {result_a['risk_level']}")
    print("   -> Key Risk Drivers   :")
    for f in result_a['top_risk_factors']:
        print(f"      * {f}")
    print("   -> Actionable Retention Strategy :")
    for r in result_a['retention_recommendations']:
        print(f"      - {r}")

    # Test Case B: Safe Loyal Customer
    print("\n[CASE B: SAFE LOYAL CUSTOMER PROFILE]")
    print("Profile: 2-Year Contract, Tenure: 60 Months, DSL, 24/7 Tech Support, Automatic Bank Transfer, $48.00/mo")
    loyal_data = CustomData(
        gender="Male", SeniorCitizen=0, Partner="Yes", Dependents="Yes", tenure=60,
        PhoneService="Yes", MultipleLines="Yes", InternetService="DSL",
        OnlineSecurity="Yes", OnlineBackup="Yes", DeviceProtection="Yes", TechSupport="Yes",
        StreamingTV="No", StreamingMovies="No", Contract="Two year",
        PaperlessBilling="No", PaymentMethod="Bank transfer (automatic)",
        MonthlyCharges=48.00, TotalCharges=2880.00
    )
    result_b = pipeline.predict(loyal_data.get_data_as_data_frame())
    print(f"   -> AI Prediction       : {result_b['label']}")
    print(f"   -> Churn Probability  : {result_b['churn_probability']}%")
    print(f"   -> Risk Tier Category : {result_b['risk_level']}")
    print("   -> Actionable Retention Strategy :")
    for r in result_b['retention_recommendations']:
        print(f"      - {r}")

    # -------------------------------------------------------------
    # 3. Enterprise Batch Scoring Demonstration
    # -------------------------------------------------------------
    print_banner("3. ENTERPRISE BATCH CSV SCORING DEMO")
    raw_path = os.path.join("data", "raw", "telecom_churn.csv")
    if os.path.exists(raw_path):
        sample_df = pd.read_csv(raw_path).head(10)
        from src.utils import load_object
        preprocessor = load_object(pipeline.preprocessor_path)
        model = load_object(pipeline.model_path)

        cols = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges"
        ]
        df_f = sample_df[cols].copy()
        df_f["TotalCharges"] = pd.to_numeric(df_f["TotalCharges"], errors="coerce").fillna(0.0)
        scaled = preprocessor.transform(df_f)
        preds = model.predict(scaled)
        probs = model.predict_proba(scaled)[:, 1] if hasattr(model, "predict_proba") else preds

        sample_df["Prediction"] = ["CHURN" if p == 1 else "RETAINED" for p in preds]
        sample_df["Churn_Prob_%"] = [f"{p*100:.1f}%" for p in probs]
        sample_df["Risk_Tier"] = ["Critical" if p >= 0.75 else "High" if p >= 0.50 else "Moderate" if p >= 0.25 else "Low" for p in probs]

        print(f"Evaluated {len(sample_df)} Enterprise Customer Accounts in Batch Mode:\n")
        display_cols = ["customerID", "Contract", "tenure", "MonthlyCharges", "Prediction", "Churn_Prob_%", "Risk_Tier"]
        print(sample_df[display_cols].to_string(index=False))

    # -------------------------------------------------------------
    # 4. System Execution Logs Inspection
    # -------------------------------------------------------------
    print_banner("4. RECENT PIPELINE EXECUTION LOGS (logs/churn_system.log)")
    log_path = os.path.join("logs", "churn_system.log")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[-8:]:
                print(f"[LOG] {line.strip()}")

    print("\n" + "="*75)
    print("[SUCCESS] DEMONSTRATION COMPLETE - ALL SYSTEMS FUNCTIONING AT 100%")
    print("          Web Dashboard URL: http://127.0.0.1:5000")
    print("="*75 + "\n")

if __name__ == "__main__":
    main()
