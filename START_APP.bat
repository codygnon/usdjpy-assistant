@echo off
REM ============================================================
REM  USDJPY Assistant - Start Application
REM  This script starts the trading assistant web app.
REM ============================================================

echo.
echo ============================================
echo   USDJPY Assistant - Starting...
echo ============================================
echo.

cd /d "%~dp0"

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Please run SETUP_ENV.bat first to create the environment.
    echo.
    pause
    exit /b 1
)

echo Starting the web application...
echo.
echo Once started, open your browser to:
echo   http://127.0.0.1:8000
echo.
echo Press Ctrl+C to stop the server.
echo.

REM Start the API server (will open browser automatically)
.\.venv\Scripts\python.exe run_api.py

pause
