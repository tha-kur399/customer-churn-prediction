"""
=============================================================================
Script: generate_dataset.py
Description: Generates the industry-standard Telecom Customer Churn dataset
with 7,043 customer records and realistic behavioral distributions
(e.g., higher churn in month-to-month contracts, electronic checks, no tech support).
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
from src.logger import logger
from src.exception import CustomException

def generate_telecom_data(num_records: int = 7043, random_state: int = 42) -> pd.DataFrame:
    """
    Creates a realistic Telecom Customer Churn dataset benchmarked
    against the standard IBM Telco Churn repository.
    """
    np.random.seed(random_state)
    logger.info(f"Generating synthetic benchmark dataset with {num_records} records.")

    # 1. Customer IDs
    customer_ids = [f"{np.random.randint(1000, 9999)}-{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}" for _ in range(num_records)]

    # 2. Demographic Features
    gender = np.random.choice(["Male", "Female"], size=num_records, p=[0.505, 0.495])
    senior_citizen = np.random.choice([0, 1], size=num_records, p=[0.838, 0.162])
    partner = np.random.choice(["Yes", "No"], size=num_records, p=[0.483, 0.517])
    
    # Dependents dependent on partner
    dependents = []
    for p in partner:
        if p == "Yes":
            dependents.append(np.random.choice(["Yes", "No"], p=[0.52, 0.48]))
        else:
            dependents.append(np.random.choice(["Yes", "No"], p=[0.10, 0.90]))
    dependents = np.array(dependents)

    # 3. Tenure (months)
    # Mixture of new customers (1-12 mo), medium (13-48 mo), and loyal (49-72 mo)
    tenure_distribution = np.random.choice([1, 2, 3], size=num_records, p=[0.35, 0.35, 0.30])
    tenure = []
    for t_type in tenure_distribution:
        if t_type == 1:
            tenure.append(np.random.randint(1, 13))
        elif t_type == 2:
            tenure.append(np.random.randint(13, 49))
        else:
            tenure.append(np.random.randint(49, 73))
    tenure = np.array(tenure)

    # 4. Phone & Internet Services
    phone_service = np.random.choice(["Yes", "No"], size=num_records, p=[0.903, 0.097])
    multiple_lines = []
    for ps in phone_service:
        if ps == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.46, 0.54]))
    multiple_lines = np.array(multiple_lines)

    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=num_records, p=[0.344, 0.440, 0.216])

    # Dependent Internet services
    online_security = []
    online_backup = []
    device_protection = []
    tech_support = []
    streaming_tv = []
    streaming_movies = []

    for iserv in internet_service:
        if iserv == "No":
            online_security.append("No internet service")
            online_backup.append("No internet service")
            device_protection.append("No internet service")
            tech_support.append("No internet service")
            streaming_tv.append("No internet service")
            streaming_movies.append("No internet service")
        else:
            online_security.append(np.random.choice(["Yes", "No"], p=[0.37, 0.63]))
            online_backup.append(np.random.choice(["Yes", "No"], p=[0.44, 0.56]))
            device_protection.append(np.random.choice(["Yes", "No"], p=[0.44, 0.56]))
            tech_support.append(np.random.choice(["Yes", "No"], p=[0.37, 0.63]))
            streaming_tv.append(np.random.choice(["Yes", "No"], p=[0.50, 0.50]))
            streaming_movies.append(np.random.choice(["Yes", "No"], p=[0.50, 0.50]))

    online_security = np.array(online_security)
    online_backup = np.array(online_backup)
    device_protection = np.array(device_protection)
    tech_support = np.array(tech_support)
    streaming_tv = np.array(streaming_tv)
    streaming_movies = np.array(streaming_movies)

    # 5. Contract, Billing & Payment
    # Contract strongly correlated with tenure
    contract = []
    for t in tenure:
        if t <= 12:
            contract.append(np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.85, 0.12, 0.03]))
        elif t <= 36:
            contract.append(np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.45, 0.40, 0.15]))
        else:
            contract.append(np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.15, 0.35, 0.50]))
    contract = np.array(contract)

    paperless_billing = np.random.choice(["Yes", "No"], size=num_records, p=[0.592, 0.408])
    payment_method = np.random.choice([
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ], size=num_records, p=[0.335, 0.229, 0.219, 0.217])

    # 6. Monthly Charges & Total Charges
    monthly_charges = []
    for i in range(num_records):
        base = 20.0
        if phone_service[i] == "Yes":
            base += 10.0
        if multiple_lines[i] == "Yes":
            base += 10.0
        if internet_service[i] == "DSL":
            base += 25.0
        elif internet_service[i] == "Fiber optic":
            base += 45.0
        
        # Add-on services
        for s in [online_security[i], online_backup[i], device_protection[i], tech_support[i], streaming_tv[i], streaming_movies[i]]:
            if s == "Yes":
                base += 5.0
        
        # Add random noise
        base += np.random.uniform(-3.5, 3.5)
        monthly_charges.append(round(max(18.25, min(118.75, base)), 2))
    monthly_charges = np.array(monthly_charges)

    # Total Charges = tenure * monthly_charges + small variance
    total_charges = []
    for i in range(num_records):
        tc = round(tenure[i] * monthly_charges[i] + np.random.uniform(-10, 10), 2)
        total_charges.append(max(monthly_charges[i], tc))
    total_charges = np.array(total_charges)

    # 7. Churn Probability Logic (Realistic Ground Truth)
    churn_list = []
    for i in range(num_records):
        score = 0.0
        # Month to month contract is #1 predictor of churn
        if contract[i] == "Month-to-month":
            score += 1.8
        elif contract[i] == "One year":
            score -= 0.5
        else:
            score -= 1.8

        # Short tenure increases churn significantly
        if tenure[i] <= 6:
            score += 1.5
        elif tenure[i] <= 18:
            score += 0.6
        elif tenure[i] >= 48:
            score -= 1.5

        # Fiber optic with high charges and no tech support has high churn
        if internet_service[i] == "Fiber optic":
            score += 0.8
        elif internet_service[i] == "No":
            score -= 1.0
            
        if tech_support[i] == "No" and internet_service[i] != "No":
            score += 0.6
        if online_security[i] == "No" and internet_service[i] != "No":
            score += 0.5
        if payment_method[i] == "Electronic check":
            score += 0.7
        if senior_citizen[i] == 1:
            score += 0.3
        if partner[i] == "No" and dependents[i] == "No":
            score += 0.3
        if monthly_charges[i] > 75.0:
            score += 0.6

        # Convert score to probability using standard sigmoid
        prob = 1.0 / (1.0 + np.exp(-(score - 0.85)))
        prob = min(0.95, max(0.02, prob))
        is_churn = "Yes" if np.random.rand() < prob else "No"
        churn_list.append(is_churn)

    # Create DataFrame
    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn_list
    })

    return df

if __name__ == "__main__":
    try:
        os.makedirs(os.path.join("data", "raw"), exist_ok=True)
        os.makedirs(os.path.join("data", "processed"), exist_ok=True)
        dataset_path = os.path.join("data", "raw", "telecom_churn.csv")
        
        df = generate_telecom_data(num_records=7043)
        df.to_csv(dataset_path, index=False)
        logger.info(f"Dataset successfully created and saved at: {dataset_path}")
        print(f"[SUCCESS] Telecom Churn dataset successfully created: {dataset_path}")
        print(f"Shape: {df.shape}")
        print(f"Churn Distribution:\n{df['Churn'].value_counts(normalize=True) * 100}")
    except Exception as e:
        raise CustomException(e, sys)
