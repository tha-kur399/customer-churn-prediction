"""
=============================================================================
Module: src/pipeline/predict_pipeline.py
Description: Inference Pipeline for Customer Churn Prediction.
Loads serialized preprocessor and trained ML model, calculates churn probability,
assigns risk tiers, identifies churn drivers, and provides actionable retention
recommendations.
=============================================================================
"""

import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logger
from src.utils import load_object


class PredictPipeline:
    """
    Inference engine that transforms raw customer feature data, executes model
    prediction, calculates risk scores, and generates retention recommendations.
    """
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.joblib")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.joblib")

    def predict(self, features: pd.DataFrame) -> dict:
        """
        Takes raw customer dataframe, performs preprocessing, and runs inference.

        Args:
            features (pd.DataFrame): Raw customer features

        Returns:
            dict: Detailed prediction results containing:
                - prediction (int: 0 or 1)
                - label (str: "Retained" or "Churn Risk")
                - churn_probability (float: 0.0 to 100.0%)
                - risk_level (str: "Low", "Moderate", "High", "Critical")
                - risk_color (str: Hex color for UI)
                - top_risk_factors (list of str)
                - retention_recommendations (list of str)
        """
        try:
            logger.info("Starting inference pipeline execution.")

            # 1. Load artifacts
            preprocessor = load_object(file_path=self.preprocessor_path)
            model = load_object(file_path=self.model_path)

            # 2. Handle TotalCharges numeric type casting
            features = features.copy()
            features["TotalCharges"] = pd.to_numeric(features["TotalCharges"], errors="coerce").fillna(0.0)

            # 3. Transform features
            logger.info("Applying preprocessor transformation to input sample.")
            data_scaled = preprocessor.transform(features)

            # 4. Predict class and probability
            preds = model.predict(data_scaled)
            pred_val = int(preds[0])

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(data_scaled)[0]
                churn_prob = float(probs[1]) * 100.0
            else:
                churn_prob = 100.0 if pred_val == 1 else 0.0

            churn_prob = round(churn_prob, 2)

            # 5. Determine Risk Category and Colors
            if churn_prob < 25.0:
                risk_level = "Low Risk"
                risk_color = "#10b981"  # Emerald Green
                status_badge = "Safe Customer"
            elif churn_prob < 50.0:
                risk_level = "Moderate Risk"
                risk_color = "#f59e0b"  # Amber
                status_badge = "Attention Recommended"
            elif churn_prob < 75.0:
                risk_level = "High Risk"
                risk_color = "#f97316"  # Orange
                status_badge = "High Churn Likelihood"
            else:
                risk_level = "Critical Risk"
                risk_color = "#ef4444"  # Red
                status_badge = "Immediate Action Required"

            # 6. Analyze customer attributes for specific Risk Drivers & Business Recommendations
            risk_factors = []
            recommendations = []

            # Check Contract
            contract_val = features["Contract"].iloc[0]
            if contract_val == "Month-to-month":
                risk_factors.append("Month-to-Month Contract: No long-term lock-in enables immediate switching.")
                recommendations.append("Offer a 15% discount on upgrading to a 1-year or 2-year annual contract plan.")

            # Check Tenure
            tenure_val = float(features["tenure"].iloc[0])
            if tenure_val <= 6:
                risk_factors.append(f"Early Lifecycle Customer ({int(tenure_val)} months tenure): Onboarding churn risk.")
                recommendations.append("Enroll customer in high-touch onboarding program with dedicated account support.")
            elif tenure_val <= 18:
                risk_factors.append(f"Young Customer Relationship ({int(tenure_val)} months tenure).")

            # Check Tech Support & Online Security
            tech_sup = features["TechSupport"].iloc[0]
            sec_val = features["OnlineSecurity"].iloc[0]
            if tech_sup == "No":
                risk_factors.append("No Tech Support Add-on: High vulnerability to technical frustration.")
                recommendations.append("Provide a complimentary 3-month trial of Premium 24/7 Tech Support.")
            if sec_val == "No":
                risk_factors.append("No Online Security: Reduced customer stickiness.")
                recommendations.append("Bundle Cybersecurity & Device Protection as a value-add perk.")

            # Check Payment Method
            pay_method = features["PaymentMethod"].iloc[0]
            if pay_method == "Electronic check":
                risk_factors.append("Electronic Check Payment: Lacks recurring auto-debit commitment.")
                recommendations.append("Incentivize automatic Bank Transfer or Credit Card setup with a $10 bill credit.")

            # Check Monthly Charges
            monthly_chg = float(features["MonthlyCharges"].iloc[0])
            if monthly_chg > 75.0:
                risk_factors.append(f"High Monthly Bill (${monthly_chg:.2f}/mo): Price sensitivity risk.")
                recommendations.append("Review plan utilization and recommend a optimized personalized tier.")

            # Default recommendations if list is short
            if not recommendations:
                recommendations.append("Maintain high customer satisfaction with periodic loyalty check-ins.")
                recommendations.append("Eligible for premium rewards and exclusive loyalty benefits.")

            logger.info(f"Inference result -> Churn Prob: {churn_prob}% | Risk Tier: {risk_level}")

            return {
                "prediction": pred_val,
                "label": "Churn Risk" if pred_val == 1 or churn_prob >= 50.0 else "Retained",
                "churn_probability": churn_prob,
                "retention_probability": round(100.0 - churn_prob, 2),
                "risk_level": risk_level,
                "risk_color": risk_color,
                "status_badge": status_badge,
                "top_risk_factors": risk_factors,
                "retention_recommendations": recommendations
            }

        except Exception as e:
            logger.error("Error occurred in PredictPipeline inference.")
            raise CustomException(e, sys)


class CustomData:
    """
    Maps individual input fields from web form or API request
    into a standardized Pandas DataFrame matching the model's schema.
    """
    def __init__(
        self,
        gender: str,
        SeniorCitizen: int,
        Partner: str,
        Dependents: str,
        tenure: float,
        PhoneService: str,
        MultipleLines: str,
        InternetService: str,
        OnlineSecurity: str,
        OnlineBackup: str,
        DeviceProtection: str,
        TechSupport: str,
        StreamingTV: str,
        StreamingMovies: str,
        Contract: str,
        PaperlessBilling: str,
        PaymentMethod: str,
        MonthlyCharges: float,
        TotalCharges: float
    ):
        self.gender = gender
        self.SeniorCitizen = int(SeniorCitizen)
        self.Partner = Partner
        self.Dependents = Dependents
        self.tenure = float(tenure)
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.MonthlyCharges = float(MonthlyCharges)
        self.TotalCharges = float(TotalCharges)

    def get_data_as_data_frame(self) -> pd.DataFrame:
        """
        Converts properties into a 1-row Pandas DataFrame.
        """
        try:
            custom_data_dict = {
                "gender": [self.gender],
                "SeniorCitizen": [self.SeniorCitizen],
                "Partner": [self.Partner],
                "Dependents": [self.Dependents],
                "tenure": [self.tenure],
                "PhoneService": [self.PhoneService],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "OnlineBackup": [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges]
            }

            return pd.DataFrame(custom_data_dict)

        except Exception as e:
            logger.error("Error creating DataFrame from CustomData.")
            raise CustomException(e, sys)
