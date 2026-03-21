@echo off
REM LS-DYNA Post-Processor UI Launcher
REM This script launches the Streamlit web interface

echo ================================
echo LS-DYNA Post-Processor
echo ================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ or add Python to your system PATH
    pause
    exit /b 1
)

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Streamlit is not installed
    echo Installing required packages from requirements.txt...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo.
echo Starting LS-DYNA Post-Processor UI...
echo The application will open in your default browser
echo Press Ctrl+C to stop the server
echo.

REM Run the Streamlit app
streamlit run LS_Post_UI.py

pause
