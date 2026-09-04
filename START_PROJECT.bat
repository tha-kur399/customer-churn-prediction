@echo off
title Customer Churn Prediction System - Web Application Launcher
color 0B
cls
echo ===============================================================================
echo            CUSTOMER CHURN PREDICTION & RETENTION SYSTEM
echo             MCA 3rd Semester Data Science Project Launcher
echo ===============================================================================
echo.
echo [*] Checking Python environment...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10+ from python.org
    pause
    exit /b
)

echo [*] Python detected successfully.
echo [*] Starting Web Application on http://127.0.0.1:5000 ...
echo [*] Opening your web browser automatically...
echo.

:: Launch browser after 2 second delay
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:5000"

:: Start Flask application
python webapp/app.py

pause
