"""
=============================================================================
Module: webapp/app.py
Description: Flask Web Application for Customer Churn Prediction System.
Provides interactive UI for:
1. Single Customer Churn Risk Prediction & Retention Recommendations
2. Batch CSV Upload & Scoring with downloadable results
3. Exploratory Data Analysis & Model Performance Insights Dashboard
4. Real-time Project Logs Viewer for College Viva Demonstration
5. REST API for programmatic predictions
=============================================================================
"""

import os
import sys
import json
import io
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.logger import logger, MASTER_LOG_PATH
from src.exception import CustomException
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)
app.config["SECRET_KEY"] = "churn-prediction-mca-project-secret-2026"
app.config["UPLOAD_FOLDER"] = os.path.join("webapp", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Cache for batch results
LATEST_BATCH_CSV_PATH = os.path.join(app.config["UPLOAD_FOLDER"], "latest_batch_results.csv")


def load_metrics_data():
    """Loads metrics.json if available."""
    metrics_path = os.path.join("artifacts", "metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read metrics.json: {e}")
    return None


@app.route("/")
def index():
    """Landing and Overview Dashboard."""
    metrics = load_metrics_data()
    return render_template("index.html", metrics=metrics)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Single Customer Risk Evaluation Form & Prediction Outcome.
    """
    if request.method == "GET":
        return render_template("predict.html", result=None)

    try:
        # Extract inputs from form
        data = CustomData(
            gender=request.form.get("gender", "Male"),
            SeniorCitizen=int(request.form.get("SeniorCitizen", 0)),
            Partner=request.form.get("Partner", "No"),
            Dependents=request.form.get("Dependents", "No"),
            tenure=float(request.form.get("tenure", 12)),
            PhoneService=request.form.get("PhoneService", "Yes"),
            MultipleLines=request.form.get("MultipleLines", "No"),
            InternetService=request.form.get("InternetService", "Fiber optic"),
            OnlineSecurity=request.form.get("OnlineSecurity", "No"),
            OnlineBackup=request.form.get("OnlineBackup", "No"),
            DeviceProtection=request.form.get("DeviceProtection", "No"),
            TechSupport=request.form.get("TechSupport", "No"),
            StreamingTV=request.form.get("StreamingTV", "No"),
            StreamingMovies=request.form.get("StreamingMovies", "No"),
            Contract=request.form.get("Contract", "Month-to-month"),
            PaperlessBilling=request.form.get("PaperlessBilling", "Yes"),
            PaymentMethod=request.form.get("PaymentMethod", "Electronic check"),
            MonthlyCharges=float(request.form.get("MonthlyCharges", 70.0)),
            TotalCharges=float(request.form.get("TotalCharges", 840.0))
        )

        pred_df = data.get_data_as_data_frame()
        logger.info(f"Form submission received for prediction: {pred_df.to_dict(orient='records')}")

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)

        # Store form inputs to preserve form state in UI
        form_data = request.form.to_dict()

        return render_template("predict.html", result=result, form_data=form_data)

    except Exception as e:
        logger.error(f"Error during single prediction: {str(e)}")
        return render_template("predict.html", error=f"Prediction Error: {str(e)}", result=None)


@app.route("/batch", methods=["GET", "POST"])
def batch_predict():
    """
    Upload CSV for batch customer churn evaluation.
    """
    if request.method == "GET":
        return render_template("batch.html", batch_summary=None)

    try:
        if "file" not in request.files:
            return render_template("batch.html", error="Please select a CSV file to upload.")

        file = request.files["file"]
        if file.filename == "":
            return render_template("batch.html", error="No file selected.")

        if not file.filename.endswith(".csv"):
            return render_template("batch.html", error="Invalid file format! Please upload a .csv file.")

        # Read uploaded CSV
        df = pd.read_csv(file)
        logger.info(f"Batch prediction initiated for uploaded file with {len(df)} records.")

        required_cols = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges"
        ]

        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return render_template(
                "batch.html",
                error=f"Uploaded CSV is missing required columns: {', '.join(missing)}"
            )

        predict_pipeline = PredictPipeline()
        # Load artifacts directly for fast vectorized batch inference
        from src.utils import load_object
        preprocessor = load_object(predict_pipeline.preprocessor_path)
        model = load_object(predict_pipeline.model_path)

        df_features = df[required_cols].copy()
        df_features["TotalCharges"] = pd.to_numeric(df_features["TotalCharges"], errors="coerce").fillna(0.0)

        data_scaled = preprocessor.transform(df_features)
        preds = model.predict(data_scaled)
        probs = model.predict_proba(data_scaled)[:, 1] if hasattr(model, "predict_proba") else preds

        df["Churn_Prediction"] = ["Churn" if p == 1 else "Retained" for p in preds]
        df["Churn_Probability_%"] = [round(p * 100, 2) for p in probs]
        df["Risk_Level"] = [
            "Critical" if p >= 75 else "High" if p >= 50 else "Moderate" if p >= 25 else "Low"
            for p in df["Churn_Probability_%"]
        ]

        # Save annotated batch file
        df.to_csv(LATEST_BATCH_CSV_PATH, index=False)

        # Compute summary metrics
        total_customers = len(df)
        churn_count = int((df["Churn_Prediction"] == "Churn").sum())
        retained_count = total_customers - churn_count
        churn_rate = round((churn_count / total_customers) * 100, 2)

        high_risk_count = int((df["Risk_Level"].isin(["High", "Critical"])).sum())
        avg_probability = round(df["Churn_Probability_%"].mean(), 2)

        batch_summary = {
            "total_customers": total_customers,
            "churn_count": churn_count,
            "retained_count": retained_count,
            "churn_rate": churn_rate,
            "high_risk_count": high_risk_count,
            "avg_probability": avg_probability,
            "sample_records": df.head(15).to_dict(orient="records")
        }

        return render_template("batch.html", batch_summary=batch_summary)

    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        return render_template("batch.html", error=f"Batch Processing Error: {str(e)}")


@app.route("/download-batch")
def download_batch():
    """Allows downloading the scored batch CSV."""
    if os.path.exists(LATEST_BATCH_CSV_PATH):
        return send_file(
            LATEST_BATCH_CSV_PATH,
            as_attachment=True,
            download_name="customer_churn_predictions_scored.csv",
            mimetype="text/csv"
        )
    return redirect(url_for("batch_predict"))


@app.route("/sample-batch")
def sample_batch():
    """Generates and downloads a ready-to-test 25-row sample CSV."""
    raw_path = os.path.join("data", "raw", "telecom_churn.csv")
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path).head(25)
        # Drop true target column to simulate unseen customer file
        if "Churn" in df.columns:
            df = df.drop(columns=["Churn"])
        
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return send_file(
            io.BytesIO(buffer.getvalue().encode()),
            as_attachment=True,
            download_name="sample_customers_test.csv",
            mimetype="text/csv"
        )
    return redirect(url_for("batch_predict"))


@app.route("/insights")
def insights():
    """Analytics & Model Performance Dashboard."""
    metrics = load_metrics_data()
    return render_template("insights.html", metrics=metrics)


@app.route("/logs")
def logs():
    """Real-time Log Viewer page."""
    log_lines = []
    if os.path.exists(MASTER_LOG_PATH):
        try:
            with open(MASTER_LOG_PATH, "r", encoding="utf-8") as f:
                log_lines = f.readlines()[-200:]  # Last 200 lines
        except Exception as e:
            log_lines = [f"Failed to load logs: {str(e)}"]
    return render_template("logs.html", log_lines=reversed(log_lines))


@app.route("/api/logs")
def api_logs():
    """API endpoint returning latest log entries as JSON."""
    if os.path.exists(MASTER_LOG_PATH):
        try:
            with open(MASTER_LOG_PATH, "r", encoding="utf-8") as f:
                lines = f.readlines()[-100:]
                return jsonify({"status": "success", "logs": lines})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify({"status": "success", "logs": []})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint for programmatic single prediction."""
    try:
        data_json = request.get_json(force=True)
        pred_df = pd.DataFrame([data_json])
        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    logger.info("Starting Flask Web Application on http://127.0.0.1:5000")
    print("\n" + "="*60)
    print("[SERVER] FLASK APP RUNNING AT: http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
