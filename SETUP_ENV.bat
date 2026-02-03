@echo off
setlocal enabledelayedexpansion

REM One-time setup: create venv + install dependencies.

cd /d "%~dp0"

if not exist "requirements.txt" (
  echo Missing requirements.txt
  exit /b 1
)

if exist ".venv\Scripts\python.exe" (
  echo .venv already exists.
  goto :install
)

echo Creating venv in .venv ...
python -m venv .venv
if errorlevel 1 (
  echo Failed to create venv. Make sure Python 3.11+ is installed.
  exit /b 1
)

:install
echo Installing dependencies...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo ============================================
echo   Setup complete!
echo ============================================
echo.
echo Next steps:
echo   1) Open MetaTrader 5 and log into a demo account
echo   2) Enable "Algo Trading" in MT5
echo.
echo To run the NEW web app (recommended):
echo   Run START_APP.bat
echo.
echo To run the old Streamlit UI:
echo   Run RUN_UI.bat
echo.
echo To run the trading loop separately:
echo   Run RUN_LOOP.bat
echo.
pause

