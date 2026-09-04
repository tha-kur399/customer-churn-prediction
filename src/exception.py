"""
=============================================================================
Module: src/exception.py
Description: Custom Exception Handler for Customer Churn Prediction System.
Captures exact file name, line number, and error message during runtime
exceptions to ensure high debuggability and professional standard in viva.
=============================================================================
"""

import sys
from src.logger import logger


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Extracts detailed execution traceback info including script name,
    line number, and the exact exception string.

    Args:
        error (Exception): The raw caught exception.
        error_detail (sys): The sys module to inspect current traceback.

    Returns:
        str: Formatted, human-readable error description.
    """
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = f"Error occurred in Python script: [{file_name}] at line number: [{line_number}] -> Error Message: [{str(error)}]"
    else:
        error_message = f"Error occurred -> Error Message: [{str(error)}]"
    return error_message


class CustomException(Exception):
    """
    Custom Exception class that overrides default string representation
    with detailed error tracking metadata.
    """
    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        logger.error(self.error_message)

    def __str__(self) -> str:
        return self.error_message


if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        custom_exc = CustomException(e, sys)
        print(custom_exc)
