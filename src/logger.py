"""
=============================================================================
Module: src/logger.py
Description: Custom Logging Configuration for Customer Churn Prediction System.
Maintains comprehensive execution logs with timestamps, file names, line numbers,
and log severity levels in both the log file (logs/churn_system.log) and console.
=============================================================================
"""

import logging
import os
import sys
from datetime import datetime

# Define directory for storing log files
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Generate a log file name based on current date
LOG_FILE_NAME = f"churn_system_{datetime.now().strftime('%Y_%m_%d')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

# Also maintain a master log file for easy access by web viewer
MASTER_LOG_PATH = os.path.join(LOG_DIR, "churn_system.log")

# Standard format for logging records
LOG_FORMAT = "[%(asctime)s] %(levelname)s | Line: %(lineno)d | Module: %(module)s.%(funcName)s() -> %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Configure root logger handlers
file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

master_file_handler = logging.FileHandler(MASTER_LOG_PATH, encoding="utf-8")
master_file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, master_file_handler, console_handler]
)

# Export standard logger
logger = logging.getLogger("ChurnPredictionSystem")

if __name__ == "__main__":
    logger.info("Custom Logger initialized successfully.")
    logger.warning("Sample warning log for testing purposes.")
