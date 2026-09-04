"""
=============================================================================
Script: test_app.py
Description: Automated test suite for Customer Churn Prediction System.
Tests data transformation, model loading, single inference, batch inference,
and Flask app endpoint availability.
=============================================================================
"""

import os
import sys
import unittest
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from webapp.app import app

class TestChurnSystem(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.predict_pipeline = PredictPipeline()

    def test_single_prediction_high_risk(self):
        """Test high risk customer profile produces high churn probability."""
        data = CustomData(
            gender="Female",
            SeniorCitizen=1,
            Partner="No",
            Dependents="No",
            tenure=2,
            PhoneService="Yes",
            MultipleLines="Yes",
            InternetService="Fiber optic",
            OnlineSecurity="No",
            OnlineBackup="No",
            DeviceProtection="No",
            TechSupport="No",
            StreamingTV="Yes",
            StreamingMovies="Yes",
            Contract="Month-to-month",
            PaperlessBilling="Yes",
            PaymentMethod="Electronic check",
            MonthlyCharges=98.50,
            TotalCharges=197.00
        )
        df = data.get_data_as_data_frame()
        res = self.predict_pipeline.predict(df)
        print(f"\n[TEST HIGH RISK] Churn Prob: {res['churn_probability']}% | Tier: {res['risk_level']}")
        self.assertIn("churn_probability", res)
        self.assertGreater(res["churn_probability"], 50.0)

    def test_single_prediction_loyal_customer(self):
        """Test loyal customer profile produces low churn probability."""
        data = CustomData(
            gender="Male",
            SeniorCitizen=0,
            Partner="Yes",
            Dependents="Yes",
            tenure=60,
            PhoneService="Yes",
            MultipleLines="Yes",
            InternetService="DSL",
            OnlineSecurity="Yes",
            OnlineBackup="Yes",
            DeviceProtection="Yes",
            TechSupport="Yes",
            StreamingTV="No",
            StreamingMovies="No",
            Contract="Two year",
            PaperlessBilling="No",
            PaymentMethod="Bank transfer (automatic)",
            MonthlyCharges=45.00,
            TotalCharges=2700.00
        )
        df = data.get_data_as_data_frame()
        res = self.predict_pipeline.predict(df)
        print(f"[TEST LOYAL] Churn Prob: {res['churn_probability']}% | Tier: {res['risk_level']}")
        self.assertIn("churn_probability", res)
        self.assertLess(res["churn_probability"], 35.0)

    def test_web_routes(self):
        """Test all web endpoints return HTTP 200."""
        routes = ["/", "/predict", "/batch", "/insights", "/logs", "/api/logs"]
        for route in routes:
            response = self.app.get(route)
            print(f"[ROUTE TEST] {route:<15} -> Status: {response.status_code}")
            self.assertEqual(response.status_code, 200)

if __name__ == "__main__":
    unittest.main()
