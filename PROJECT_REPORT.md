# Customer Churn Prediction & Retention Optimization System
**Academic Project Report — Master of Computer Applications (MCA 3rd Semester)**  
**Specialization:** Data Science & Machine Learning  

---

## Table of Contents
1. [Project Synopsis & Abstract](#1-project-synopsis--abstract)
2. [Problem Statement & Objectives](#2-problem-statement--objectives)
3. [System Architecture & Lifecycle](#3-system-architecture--lifecycle)
4. [Data Dictionary & Feature Schema](#4-data-dictionary--feature-schema)
5. [Data Preprocessing & Feature Engineering](#5-data-preprocessing--feature-engineering)
6. [Machine Learning Algorithms Benchmarked](#6-machine-learning-algorithms-benchmarked)
7. [Model Evaluation & Comparative Analysis](#7-model-evaluation--comparative-analysis)
8. [Business Value & Prescriptive Retention Actions](#8-business-value--prescriptive-retention-actions)
9. [Web Application & Deployment Design](#9-web-application--deployment-design)
10. [Top 25 College Viva Questions & Model Answers](#10-top-25-college-viva-questions--model-answers)

---

## 1. Project Synopsis & Abstract

Customer Churn (attrition) is one of the most critical challenges facing the telecom and subscription services industry. Acquiring a new customer costs 5 to 7 times more than retaining an existing one. 

This project implements an **End-to-End, Production-Grade Machine Learning System** to predict customer churn probability before cancellation occurs. The system benchmarks multiple supervised learning algorithms (Logistic Regression, Random Forest, Decision Tree, Gradient Boosting, AdaBoost), evaluates them using stratified holdout validation, packages the best preprocessor and model into serialized pipelines, and deploys an interactive Flask web application equipped with:
- Real-time customer churn risk scoring and animated gauge.
- Root-cause churn driver detection.
- Actionable, personalized retention recommendations.
- Enterprise CSV batch scoring with instant export.
- Visual Exploratory Data Analysis (EDA) and model performance dashboards.
- Real-time rotating file and console logging system.

---

## 2. Problem Statement & Objectives

### Problem Statement
Subscription-based businesses frequently suffer revenue losses due to unpredicted customer churn. Traditional rule-based alerts fail to detect complex non-linear churn patterns across demographics, contract terms, support interactions, and payment behaviors.

### Objectives
1. **Develop an automated ML Pipeline**: Clean, transform, and encode multi-modal customer data without data leakage.
2. **Benchmark Classifiers**: Train multiple models and compare Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
3. **Handle Imbalanced Data**: Utilize stratified sampling and cost-sensitive class balancing.
4. **Deploy Usable Software**: Provide a web-based portal for single customer profiling and enterprise batch CSV scoring.
5. **Ensure Code Quality & Traceability**: Maintain structured timestamped logging (`logs/churn_system.log`) and custom exception tracking.

---

## 3. System Architecture & Lifecycle

```
[Raw Dataset: 7,043 Records]
            │
            ▼
┌───────────────────────────────────────────────┐
│ 1. Data Ingestion (data_ingestion.py)         │
│    - Schema validation & null integrity       │
│    - 80:20 Stratified Train-Test Partition    │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│ 2. Data Transformation (data_transformation)  │
│    - Numerical: Median Imputer + StandardScaler│
│    - Categorical: Mode Imputer + OneHotEncoder│
│    - ColumnTransformer Serialization          │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│ 3. Model Benchmarking (model_trainer.py)      │
│    - Logistic Regression, Random Forest,      │
│      Gradient Boosting, AdaBoost, Decision Tree│
│    - Evaluation: Acc, F1, Precision, Recall,  │
│      ROC-AUC Curve, Confusion Matrix          │
│    - Best Model Serialized (model.joblib)     │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│ 4. Deployment & Web Portal (webapp/app.py)    │
│    - Single Assessment + Business Actions     │
│    - CSV Batch Prediction & Export            │
│    - EDA Visualizations & Live Logs Stream    │
└───────────────────────────────────────────────┘
```

---

## 4. Data Dictionary & Feature Schema

The dataset comprises **7,043 customer accounts** with **21 attributes**:

| Feature Name | Data Type | Value Range / Categories | Description |
|---|---|---|---|
| `customerID` | String | Unique Alphanumeric | Unique customer identifier |
| `gender` | Categorical | Male, Female | Customer gender |
| `SeniorCitizen` | Binary | 0 (No), 1 (Yes) | Whether the customer is a senior citizen |
| `Partner` | Categorical | Yes, No | Customer has a partner/spouse |
| `Dependents` | Categorical | Yes, No | Customer has financial dependents |
| `tenure` | Numeric | 1 to 72 (months) | Number of months the customer has stayed |
| `PhoneService` | Categorical | Yes, No | Subscribed to landline phone service |
| `MultipleLines` | Categorical | Yes, No, No phone service | Multiple telephone lines |
| `InternetService` | Categorical | DSL, Fiber optic, No | Type of internet service provider |
| `OnlineSecurity` | Categorical | Yes, No, No internet | Cybersecurity add-on subscription |
| `OnlineBackup` | Categorical | Yes, No, No internet | Cloud storage backup add-on |
| `DeviceProtection`| Categorical | Yes, No, No internet | Hardware warranty protection add-on |
| `TechSupport` | Categorical | Yes, No, No internet | 24/7 dedicated technical support |
| `StreamingTV` | Categorical | Yes, No, No internet | Streaming television package |
| `StreamingMovies`| Categorical | Yes, No, No internet | Streaming movie package |
| `Contract` | Categorical | Month-to-month, 1 yr, 2 yr | Terms of current contract |
| `PaperlessBilling`| Categorical| Yes, No | Opted for electronic billing |
| `PaymentMethod` | Categorical | Electronic check, Mailed check, Bank transfer, Credit card | Method used for bill payment |
| `MonthlyCharges`| Numeric | $18.25 - $118.75 | Current monthly invoice amount |
| `TotalCharges` | Numeric | $18.25 - $8684.80 | Total cumulative charges incurred |
| `Churn` (Target)| Categorical | Yes (1), No (0) | Customer churn status (Ground Truth) |

---

## 5. Data Preprocessing & Feature Engineering

1. **Type Casting & Missing Value Handling**:
   - `TotalCharges` is cast to float numeric, with empty records imputed using median strategy.
2. **Feature Separation**:
   - **Numerical Features** ($3$): `tenure`, `MonthlyCharges`, `TotalCharges`.
   - **Categorical Features** ($16$): `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`.
3. **Pipeline Construction**:
   - Numerical sub-pipeline: `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
     $$z = \frac{x - \mu}{\sigma}$$
   - Categorical sub-pipeline: `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
4. **Data Leakage Prevention**:
   - The preprocessor pipeline is fitted strictly on the Training set (`fit_transform`) and subsequently applied to the Test set and live inference samples using (`transform`).

---

## 6. Machine Learning Algorithms Benchmarked

1. **Logistic Regression (Class-Weighted)**:
   - Sigmoid activation mapping linear combinations of features to probability intervals $[0, 1]$:
     $$P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \sum \beta_i X_i)}}$$
2. **Decision Tree Classifier**:
   - Non-linear rule partition based on Gini Impurity reduction:
     $$Gini(D) = 1 - \sum_{i=1}^{k} p_i^2$$
3. **Random Forest Classifier**:
   - Ensemble of decorrelated decision trees using bootstrap aggregating (bagging) and random feature subspace sampling.
4. **Gradient Boosting Classifier**:
   - Sequential boosting ensemble fitting subsequent trees on the pseudo-residuals of prior estimators using gradient descent.
5. **AdaBoost Classifier**:
   - Adaptive boosting adjusting sample instance weights focusing subsequent weak learners on previously misclassified records.

---

## 7. Model Evaluation & Comparative Analysis

Evaluation executed on a **1,409 record stratified holdout test set**:

| Algorithm | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression (Champion)** | **81.90%** | **0.8409** | **0.8409** | **0.8409** | **0.9033** |
| **AdaBoost Classifier** | 81.33% | 0.8417 | 0.8417 | 0.8417 | 0.8991 |
| **Random Forest Classifier** | 81.62% | 0.8406 | 0.8406 | 0.8406 | 0.8980 |
| **Gradient Boosting Classifier** | 80.98% | 0.8395 | 0.8395 | 0.8395 | 0.8977 |
| **Decision Tree Classifier** | 80.77% | 0.8324 | 0.8324 | 0.8324 | 0.8756 |

### Key Findings:
- **Logistic Regression** and **AdaBoost** demonstrated superior ROC-AUC scores ($\ge 0.90$) and balanced precision/recall trade-offs.
- The highest contributing features driving churn were:
  1. `Contract_Month-to-month` (Strongest positive correlation with churn)
  2. `tenure` (Strong negative correlation: longer tenure = exponentially lower churn)
  3. `InternetService_Fiber optic` combined with no `TechSupport`
  4. `PaymentMethod_Electronic check`

---

## 8. Business Value & Prescriptive Retention Actions

The prediction pipeline does not simply return a binary classification ($0$ or $1$); it provides prescriptive business intelligence:

| Churn Probability | Risk Tier | Prescriptive Retention Actions |
|---|---|---|
| **0% – 25%** | **Low Risk** | Standard loyalty perks, referral discount program enrollment. |
| **25% – 50%** | **Moderate Risk** | Proactive customer service check-in, annual plan renewal incentives. |
| **50% – 75%** | **High Risk** | Offer 15% discount on 1-year contract lock-in, free 3-month Tech Support trial. |
| **75% – 100%** | **Critical Risk** | Dedicated account manager outreach, waive fee for auto-pay switch, immediate billing review. |

---

## 9. Web Application & Deployment Design

Built using **Flask**, **HTML5/CSS3** (Glassmorphism Dark UI), and **Vanilla JS**:
- `GET/POST /predict`: Interactive form with 1-click presets (High Risk, Loyal, Moderate).
- `GET/POST /batch`: Multi-row CSV file ingestion, vectorized model scoring, risk tiering, and CSV download.
- `GET /insights`: Interactive visual dashboard rendering Confusion Matrix, ROC-AUC, Feature Importance, and EDA plots.
- `GET /logs` & `/api/logs`: Real-time streaming log terminal with auto-polling for viva demonstration.
- `POST /api/predict`: JSON API endpoint for programmatic integration.

---

## 10. Top 25 College Viva Questions & Model Answers

### Q1: What is Customer Churn and why is predicting it important?
> **Answer:** Customer churn refers to when an existing subscriber cancels their service or subscription. Predicting churn is crucial because customer acquisition costs (CAC) are typically 5 to 7 times higher than customer retention costs. Early identification allows businesses to take targeted retention actions before the customer leaves.

### Q2: What is Data Leakage and how did you prevent it in your pipeline?
> **Answer:** Data leakage occurs when information from the test dataset is mistakenly used during model training or preprocessing, leading to overly optimistic evaluation metrics. We prevented this by first splitting the data into train and test sets, fitting our `ColumnTransformer` only on the training set (`fit_transform`), and then transforming the test set (`transform`).

### Q3: Why did you use `ColumnTransformer` instead of manual pandas transformations?
> **Answer:** `ColumnTransformer` allows applying different preprocessing pipelines (e.g. `StandardScaler` for numeric columns and `OneHotEncoder` for categorical columns) in a single unified Scikit-learn object. This object can be saved with `joblib` and directly used for inference, ensuring reproducible transformations on new data.

### Q4: Why did you use Stratified Train-Test Split?
> **Answer:** Customer churn datasets typically exhibit class imbalance (e.g., ~26% churned vs ~74% retained). Stratified splitting ensures that the proportion of churned to retained customers is identical in both the training set (80%) and the testing set (20%).

### Q5: Why is Accuracy alone not an adequate metric for churn prediction?
> **Answer:** In imbalanced datasets, a naive model predicting "No Churn" for every customer would still achieve ~75% accuracy while failing completely at identifying churners. Therefore, metrics like **Precision**, **Recall**, **F1-Score**, and **ROC-AUC** provide a more balanced evaluation of classifier performance.

### Q6: What is the difference between Precision and Recall in churn prediction?
> **Answer:** 
> - **Precision:** Out of all customers predicted to churn, how many actually churned? (High precision minimizes wasted retention discounts on non-churners).
> - **Recall (Sensitivity):** Out of all customers who actually churned, how many did the model identify? (High recall ensures we do not miss customers who are about to leave).

### Q7: What does the ROC-AUC score represent?
> **Answer:** The Receiver Operating Characteristic - Area Under the Curve (ROC-AUC) measures the model's ability to discriminate between positive (churn) and negative (retained) classes across all possible classification probability thresholds. A score of 0.5 is random guessing, while 1.0 represents perfect discrimination. Our model achieves ~0.903.

### Q8: How does Logistic Regression calculate churn probability?
> **Answer:** Logistic Regression computes a weighted linear combination of input features $z = \beta_0 + \sum \beta_i x_i$ and passes it through the Sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$, compressing the output to a probability value between 0 and 1.

### Q9: What is the purpose of the `class_weight='balanced'` parameter?
> **Answer:** It automatically adjusts model weights inversely proportional to class frequencies in the input data: $w_j = \frac{N}{k \cdot N_j}$. This penalizes errors on the minority class (churners) more heavily, preventing the model from favoring the majority class.

### Q10: How does Random Forest differ from Decision Trees?
> **Answer:** A Decision Tree creates a single tree susceptible to high variance and overfitting. Random Forest builds an ensemble of multiple decorrelated decision trees using bagging (bootstrap sampling) and random feature subsets at each split, averaging their predictions to reduce variance and improve generalization.

### Q11: How does Gradient Boosting work?
> **Answer:** Gradient Boosting builds trees sequentially. Unlike Random Forest where trees are independent, each new tree in Gradient Boosting is trained to predict the residual errors (pseudo-residuals) of the combined ensemble from the previous iterations using gradient descent on the loss function.

### Q12: What were the top 3 strongest features predicting customer churn?
> **Answer:**
> 1. **Contract Type:** Month-to-month contracts had significantly higher churn risk than 1 or 2-year commitments.
> 2. **Tenure:** Customers in their first 1–6 months had the highest churn rate.
> 3. **Internet Service & Tech Support:** Customers with Fiber Optic service but lacking 24/7 Tech Support exhibited high churn.

### Q13: What is One-Hot Encoding and why did you use `handle_unknown='ignore'`?
> **Answer:** One-Hot Encoding converts categorical variables into binary dummy vectors ($0$ or $1$). The `handle_unknown='ignore'` parameter ensures that if a new or unseen category is passed during live inference, the preprocessor will output all zeros for that feature instead of crashing with an exception.

### Q14: How did you implement structured logging in this project?
> **Answer:** We configured Python's `logging` module in `src/logger.py` to write formatted logs simultaneously to the console and to `logs/churn_system.log`. Every log entry records timestamp, severity level, source line number, and function name.

### Q15: How does your custom exception handler work?
> **Answer:** In `src/exception.py`, we created `CustomException` which inspects `sys.exc_info()`, extracts the exact script filename and line number where the runtime error originated, formats a detailed error message, logs it automatically, and raises a traceable exception.

### Q16: How are model artifacts serialized and saved?
> **Answer:** We used `joblib.dump()` in `src/utils.py` to serialize the fitted `ColumnTransformer` (`artifacts/preprocessor.joblib`) and the champion model (`artifacts/model.joblib`) to disk, allowing fast deserialization with `joblib.load()` during inference.

### Q17: What is the function of `predict_proba()` in the inference pipeline?
> **Answer:** Instead of returning only a binary classification ($0$ or $1$), `predict_proba()` returns the continuous probability distribution $[P(0), P(1)]$. We use $P(1)$ to compute the churn percentage ($0\%$ to $100\%$) and assign risk tiers (Low, Moderate, High, Critical).

### Q18: How does the Batch Prediction feature work?
> **Answer:** The user uploads a CSV file containing multiple customer records. The server verifies columns, applies the preprocessor pipeline in a vectorized operation, computes predictions and risk categories for all rows, displays the top records in an interactive table, and allows downloading the fully scored CSV.

### Q19: What is the purpose of `StandardScaler` on numerical features?
> **Answer:** Features like `TotalCharges` (up to \$8,000) and `tenure` (up to 72) operate on vastly different scales. `StandardScaler` centers features to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$), preventing features with larger numerical magnitudes from dominating distance-based and gradient-based algorithms.

### Q20: What is the difference between Bagging and Boosting?
> **Answer:** 
> - **Bagging (e.g., Random Forest):** Trains multiple base models in parallel on random bootstrap subsets of the dataset to reduce variance.
> - **Boosting (e.g., Gradient Boosting, AdaBoost):** Trains models sequentially, where each model corrects the errors of its predecessor to reduce bias and variance.

### Q21: What is a Confusion Matrix?
> **Answer:** A $2 \times 2$ table that summarizes classification performance by comparing actual values with predicted values:
> - **True Negative (TN):** Correctly predicted as Retained.
> - **False Positive (FP):** Retained customer incorrectly predicted as Churned (Type I error).
> - **False Negative (FN):** Churned customer incorrectly predicted as Retained (Type II error).
> - **True Positive (TP):** Correctly predicted as Churned.

### Q22: What is the F1-Score and when should it be favored over Accuracy?
> **Answer:** The F1-Score is the harmonic mean of Precision and Recall:
> $$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
> It should be favored whenever dataset classes are imbalanced and both False Positives and False Negatives carry business costs.

### Q23: How does your web app support real-time log monitoring?
> **Answer:** The `/logs` view fetches the last 200 lines from `logs/churn_system.log`. A client-side JavaScript interval polls `/api/logs` every 3 seconds to dynamically update the live terminal display.

### Q24: How can this model be retrained in the future when new data arrives?
> **Answer:** The pipeline is fully automated. Executing `python run_pipeline.py` or triggering `TrainPipeline().run_pipeline()` automatically ingests fresh data, rebuilds preprocessors, retrains and benchmarks all algorithms, updates evaluation plots, and replaces the serialized artifacts in `artifacts/`.

### Q25: How does this project deliver actionable business value?
> **Answer:** In addition to risk scores, the engine evaluates customer attributes to pinpoint specific churn risk drivers (e.g., no tech support, electronic check, month-to-month contract) and generates tailored recommendations (e.g., contract upgrade discount, complimentary support trial, automated billing incentive) to directly support customer retention teams.
