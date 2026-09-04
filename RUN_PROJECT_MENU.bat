@echo off
title Customer Churn Prediction & Retention System - Master Control Panel
color 0B

:MENU
cls
echo ===============================================================================
echo            CUSTOMER CHURN PREDICTION & RETENTION SYSTEM
echo             MCA 3rd Semester Data Science Project Control Panel
echo ===============================================================================
echo.
echo   [1] Start Web Application Dashboard (Opens in Browser: http://127.0.0.1:5000)
echo   [2] Run Live Model Demonstration & Inferences (CLI Mode)
echo   [3] Re-Train Machine Learning Models & Regenerate Benchmark Plots
echo   [4] Run Automated Unit Test Suite (Verify Pipeline & Endpoints)
echo   [5] Open Project Directory & Reports Folder in File Explorer
echo   [6] Open Academic Project Report & Top 25 Viva Q&A (Markdown)
echo   [7] Exit
echo.
echo ===============================================================================
set /p choice="Select an option (1-7) and press Enter: "

if "%choice%"=="1" goto START_WEB
if "%choice%"=="2" goto RUN_DEMO
if "%choice%"=="3" goto RETRAIN
if "%choice%"=="4" goto RUN_TESTS
if "%choice%"=="5" goto OPEN_DIR
if "%choice%"=="6" goto OPEN_REPORT
if "%choice%"=="7" exit
goto MENU

:START_WEB
cls
echo Starting Web Application on http://127.0.0.1:5000 ...
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:5000"
python webapp/app.py
pause
goto MENU

:RUN_DEMO
cls
python demo_run.py
pause
goto MENU

:RETRAIN
cls
python run_pipeline.py
pause
goto MENU

:RUN_TESTS
cls
python test_app.py
pause
goto MENU

:OPEN_DIR
cls
echo Opening project directory in Windows File Explorer...
start "" "%~dp0"
goto MENU

:OPEN_REPORT
cls
start "" "%~dp0PROJECT_REPORT.md"
goto MENU
