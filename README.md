# Customer Churn Prediction & Retention System
> **MCA 3rd Semester Project (Data Science Specialization)**  
> An End-to-End Machine Learning System with Modular Pipelines, Logging, Exception Tracking, Visual Analytics, and an Interactive Web Dashboard.

---

## 🌟 Key Highlights

- **Modular Architecture**: Clean separation of concerns (`DataIngestion` -> `DataTransformation` -> `ModelTrainer` -> `TrainPipeline` / `PredictPipeline`).
- **Comprehensive Logging**: Dual file (`logs/churn_system.log`) and console logging with timestamps, line numbers, and module tracking.
- **Custom Exception Handling**: Detailed error traceback tracking capturing script file name, line numbers, and root cause.
- **Multi-Algorithm Benchmarking**: Evaluates Logistic Regression, Random Forest, Gradient Boosting, AdaBoost, and Decision Tree on Accuracy, F1-Score, Precision, Recall, and ROC-AUC.
- **Modern Web Application (Flask)**:
  - **Single Customer Risk Assessment**: Live churn probability percentage, risk tier, key risk drivers, and prescriptive retention actions.
  - **1-Click Viva Demo Presets**: Test "High Risk Customer", "Loyal Safe Customer", and "Moderate Customer" with a single click.
  - **Enterprise Batch Scoring**: Upload multi-row CSV, view scored table preview, and export results.
  - **Analytics & Model Insights Dashboard**: High-res visual charts (Confusion Matrix, ROC Curve, Feature Importance, EDA plots).
  - **Live Terminal Log Viewer**: Real-time log monitoring directly inside the browser.

---

## 📁 Directory Structure

```
college-project/
│
├── data/
│   ├── raw/
│   │   └── telecom_churn.csv        # Raw dataset (7,043 records, 21 columns)
│   └── processed/
│       ├── train.csv                # 80% Stratified Training set
│       └── test.csv                 # 20% Stratified Testing set
│
├── artifacts/
│   ├── preprocessor.joblib          # Fitted ColumnTransformer (Scaling + OneHot)
│   ├── model.joblib                 # Serialized champion model
│   └── metrics.json                 # Model benchmark results & evaluation metrics
│
├── logs/
│   └── churn_system.log             # Rotating system & pipeline execution logs
│
├── reports/                         # Visual evaluation & EDA charts (PNG)
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── eda_*.png
│
├── src/                             # Core modular package
│   ├── __init__.py
│   ├── logger.py                    # Structured logging handler
│   ├── exception.py                 # Custom error & traceback handler
│   ├── utils.py                     # Serialization & metric evaluation utilities
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py        # Data ingestion & stratified splitting
│   │   ├── data_transformation.py   # Preprocessing pipeline construction
│   │   └── model_trainer.py         # Multi-model training & plot generator
│   │
│   └── pipeline/
│       ├── __init__.py
│       ├── train_pipeline.py        # End-to-end training orchestrator
│       └── predict_pipeline.py      # Inference & retention recommendation engine
│
├── webapp/                          # Web Application
│   ├── app.py                       # Flask server endpoints
│   ├── static/
│   │   ├── css/style.css            # Modern Glassmorphism dark UI theme
│   │   ├── js/main.js               # Frontend interactions & demo presets
│   │   └── images/                  # Chart assets for web rendering
│   └── templates/
│       ├── base.html                # Master layout with navigation
│       ├── index.html               # Landing & KPI dashboard
│       ├── predict.html             # Single prediction form & animated gauge
│       ├── batch.html               # Batch CSV scoring & export
│       ├── insights.html            # Visual analytics & algorithm matrix
│       └── logs.html                # Real-time streaming log terminal
│
├── eda_analysis.py                  # Exploratory Data Analysis generator
├── generate_dataset.py              # Benchmark dataset generator
├── run_pipeline.py                  # One-click master ML pipeline script
├── test_app.py                      # Automated test suite
├── requirements.txt                 # Project dependencies
├── PROJECT_REPORT.md                # Comprehensive Project Report & Top 25 Viva Q&A
└── README.md                        # Documentation & setup guide
```

---

## 🚀 Quick Start & Execution Guide

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Run the Machine Learning Pipeline (1-Click)
This script ingests data, builds the preprocessing pipeline, benchmarks all 5 algorithms, produces visual plots, and saves model artifacts:
```bash
python run_pipeline.py
```

### 3. Run Automated Tests
```bash
python test_app.py
```

### 4. Launch the Web Application
```bash
python webapp/app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

## 📊 Model Benchmark Summary

| Algorithm | Test Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| **Logistic Regression (Champion)** | **81.90%** | **0.8409** | **0.9033** |
| **AdaBoost Classifier** | 81.33% | 0.8417 | 0.8991 |
| **Random Forest Classifier** | 81.62% | 0.8406 | 0.8980 |
| **Gradient Boosting Classifier** | 80.98% | 0.8395 | 0.8977 |
| **Decision Tree Classifier** | 80.77% | 0.8324 | 0.8756 |

---

## 🎓 College Viva Preparation
Read [`PROJECT_REPORT.md`](file:///c:/college%20project/PROJECT_REPORT.md) for the complete academic report, architecture diagrams, algorithm mathematical foundations, and the **Top 25 College Viva Questions with detailed answers**.
