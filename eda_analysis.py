"""
=============================================================================
Script: eda_analysis.py
Description: Performs in-depth Exploratory Data Analysis (EDA) on the Customer
Churn dataset, computes summary statistics, and exports informative visual
charts for the project dashboard and academic presentation.
=============================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.logger import logger
from src.exception import CustomException

def run_eda(raw_data_path: str = "data/raw/telecom_churn.csv", output_dir: str = "reports", static_dir: str = "webapp/static/images"):
    """
    Generates EDA charts:
    1. Churn Class Distribution (Donut Chart)
    2. Churn Rate by Contract Type (Bar Chart)
    3. Churn Rate by Internet Service & Tech Support (Grouped Bar Chart)
    4. Tenure Distribution by Churn Status (KDE / Histogram)
    5. Monthly Charges vs Churn Distribution (Violin / Box Plot)
    """
    try:
        logger.info(">>> Starting Exploratory Data Analysis (EDA)...")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(static_dir, exist_ok=True)

        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"Raw dataset not found at: {raw_data_path}")

        df = pd.read_csv(raw_data_path)
        logger.info(f"Dataset shape: {df.shape}")

        plt.style.use("dark_background")
        colors = ["#38bdf8", "#f43f5e", "#10b981", "#fbbf24", "#818cf8"]

        # -------------------------------------------------------------
        # 1. Churn Class Distribution (Donut Chart)
        # -------------------------------------------------------------
        churn_counts = df["Churn"].value_counts()
        plt.figure(figsize=(6, 5), facecolor="#111827")
        plt.pie(
            churn_counts,
            labels=["Retained (No)", "Churned (Yes)"],
            autopct="%1.1f%%",
            startangle=140,
            colors=["#3b82f6", "#ef4444"],
            wedgeprops=dict(width=0.4, edgecolor="#111827", linewidth=2),
            textprops={"fontsize": 12, "color": "#f3f4f6"}
        )
        plt.title("Overall Customer Churn Distribution", fontsize=14, color="#f3f4f6", pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eda_churn_distribution.png"), dpi=200)
        plt.savefig(os.path.join(static_dir, "eda_churn_distribution.png"), dpi=200)
        plt.close()
        logger.info("Generated: eda_churn_distribution.png")

        # -------------------------------------------------------------
        # 2. Churn Rate by Contract Type
        # -------------------------------------------------------------
        contract_churn = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
        plt.figure(figsize=(7, 5), facecolor="#111827")
        ax = contract_churn.plot(
            kind="bar",
            stacked=False,
            color=["#10b981", "#ef4444"],
            figsize=(7, 5)
        )
        ax.set_facecolor("#1f2937")
        plt.gcf().patch.set_facecolor("#111827")
        plt.title("Churn Rate (%) by Contract Type", fontsize=14, color="#f3f4f6", pad=15)
        plt.xlabel("Contract Type", fontsize=12, color="#9ca3af")
        plt.ylabel("Percentage (%)", fontsize=12, color="#9ca3af")
        plt.xticks(rotation=0, color="#f3f4f6")
        plt.yticks(color="#f3f4f6")
        plt.legend(["Retained", "Churned"], facecolor="#1f2937", edgecolor="#374151", labelcolor="#f3f4f6")
        plt.grid(axis="y", alpha=0.15, color="#6b7280")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eda_contract_churn.png"), dpi=200)
        plt.savefig(os.path.join(static_dir, "eda_contract_churn.png"), dpi=200)
        plt.close()
        logger.info("Generated: eda_contract_churn.png")

        # -------------------------------------------------------------
        # 3. Churn by Internet Service
        # -------------------------------------------------------------
        internet_churn = pd.crosstab(df["InternetService"], df["Churn"], normalize="index") * 100
        plt.figure(figsize=(7, 5), facecolor="#111827")
        ax = internet_churn.plot(
            kind="bar",
            color=["#3b82f6", "#f97316"],
            figsize=(7, 5)
        )
        ax.set_facecolor("#1f2937")
        plt.gcf().patch.set_facecolor("#111827")
        plt.title("Churn Rate (%) by Internet Service Type", fontsize=14, color="#f3f4f6", pad=15)
        plt.xlabel("Internet Service", fontsize=12, color="#9ca3af")
        plt.ylabel("Percentage (%)", fontsize=12, color="#9ca3af")
        plt.xticks(rotation=0, color="#f3f4f6")
        plt.yticks(color="#f3f4f6")
        plt.legend(["Retained", "Churned"], facecolor="#1f2937", edgecolor="#374151", labelcolor="#f3f4f6")
        plt.grid(axis="y", alpha=0.15, color="#6b7280")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eda_internet_churn.png"), dpi=200)
        plt.savefig(os.path.join(static_dir, "eda_internet_churn.png"), dpi=200)
        plt.close()
        logger.info("Generated: eda_internet_churn.png")

        # -------------------------------------------------------------
        # 4. Tenure vs Monthly Charges Scatter / KDE
        # -------------------------------------------------------------
        plt.figure(figsize=(8, 5), facecolor="#111827")
        ax = plt.subplot(1, 1, 1)
        ax.set_facecolor("#1f2937")
        sns.histplot(
            data=df,
            x="tenure",
            hue="Churn",
            multiple="stack",
            palette={"No": "#3b82f6", "Yes": "#ef4444"},
            bins=25,
            edgecolor="#111827"
        )
        plt.title("Tenure Distribution by Churn Status", fontsize=14, color="#f3f4f6", pad=15)
        plt.xlabel("Customer Tenure (Months)", fontsize=12, color="#9ca3af")
        plt.ylabel("Customer Count", fontsize=12, color="#9ca3af")
        plt.grid(axis="y", alpha=0.15, color="#6b7280")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eda_tenure_distribution.png"), dpi=200)
        plt.savefig(os.path.join(static_dir, "eda_tenure_distribution.png"), dpi=200)
        plt.close()
        logger.info("Generated: eda_tenure_distribution.png")

        # -------------------------------------------------------------
        # 5. Monthly Charges Boxplot by Churn
        # -------------------------------------------------------------
        plt.figure(figsize=(7, 5), facecolor="#111827")
        ax = plt.subplot(1, 1, 1)
        ax.set_facecolor("#1f2937")
        sns.boxplot(
            data=df,
            x="Churn",
            y="MonthlyCharges",
            palette={"No": "#10b981", "Yes": "#f43f5e"},
            boxprops=dict(edgecolor="#f3f4f6"),
            whiskerprops=dict(color="#f3f4f6"),
            capprops=dict(color="#f3f4f6"),
            medianprops=dict(color="#ffffff", linewidth=2)
        )
        plt.title("Monthly Charges vs Churn Status", fontsize=14, color="#f3f4f6", pad=15)
        plt.xlabel("Churn Status", fontsize=12, color="#9ca3af")
        plt.ylabel("Monthly Charges ($)", fontsize=12, color="#9ca3af")
        plt.grid(axis="y", alpha=0.15, color="#6b7280")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eda_monthly_charges.png"), dpi=200)
        plt.savefig(os.path.join(static_dir, "eda_monthly_charges.png"), dpi=200)
        plt.close()
        logger.info("Generated: eda_monthly_charges.png")

        logger.info(">>> EDA Visual Charts successfully generated.")
        return True

    except Exception as e:
        logger.error("Error during EDA plot generation.")
        raise CustomException(e, sys)

if __name__ == "__main__":
    run_eda()
