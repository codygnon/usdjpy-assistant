@echo off
REM ============================================================
REM  USDJPY Assistant - Create Distribution Package
REM  This script creates a zip file for distribution.
REM ============================================================

echo.
echo ============================================
echo   Creating Distribution Package
echo ============================================
echo.

cd /d "%~dp0"

REM Check if frontend is built
if not exist "frontend\dist\index.html" (
    echo [WARNING] Frontend not built!
    echo.
    echo The distribution will work but won't have the web UI.
    echo Run BUILD_FRONTEND.bat first for the full experience.
    echo.
    choice /c YN /m "Continue anyway?"
    if errorlevel 2 exit /b 0
)

REM Create dist folder
if exist "dist" rmdir /s /q dist
mkdir dist\usdjpy_assistant

echo Copying files...

REM Copy Python code
xcopy /s /e /i "adapters" "dist\usdjpy_assistant\adapters"
xcopy /s /e /i "api" "dist\usdjpy_assistant\api"
xcopy /s /e /i "core" "dist\usdjpy_assistant\core"
xcopy /s /e /i "storage" "dist\usdjpy_assistant\storage"

REM Copy built frontend (if exists)
if exist "frontend\dist" (
    xcopy /s /e /i "frontend\dist" "dist\usdjpy_assistant\frontend\dist"
)

REM Copy profiles
xcopy /s /e /i "profiles" "dist\usdjpy_assistant\profiles"

REM Copy main scripts
copy "run_loop.py" "dist\usdjpy_assistant\"
copy "run_api.py" "dist\usdjpy_assistant\"
copy "doctor_mt5.py" "dist\usdjpy_assistant\"
copy "snapshot_log.py" "dist\usdjpy_assistant\"
copy "ui_app.py" "dist\usdjpy_assistant\"
copy "close_trade.py" "dist\usdjpy_assistant\"
copy "review_stats.py" "dist\usdjpy_assistant\"

REM Copy batch files
copy "*.bat" "dist\usdjpy_assistant\"

REM Copy config and docs
copy "requirements.txt" "dist\usdjpy_assistant\"
copy "README.md" "dist\usdjpy_assistant\"
if exist ".streamlit" (
    xcopy /s /e /i ".streamlit" "dist\usdjpy_assistant\.streamlit"
)

REM Create empty logs folder
mkdir "dist\usdjpy_assistant\logs"

REM Remove development files from dist
del /q "dist\usdjpy_assistant\CREATE_DISTRIBUTION.bat" 2>nul
del /q "dist\usdjpy_assistant\BUILD_FRONTEND.bat" 2>nul

echo.
echo Creating zip file...

REM Try to use PowerShell to create zip
powershell -command "Compress-Archive -Path 'dist\usdjpy_assistant' -DestinationPath 'dist\usdjpy_assistant.zip' -Force"

if %ERRORLEVEL% equ 0 (
    echo.
    echo ============================================
    echo   Distribution created successfully!
    echo   Location: dist\usdjpy_assistant.zip
    echo ============================================
) else (
    echo.
    echo [WARNING] Could not create zip file automatically.
    echo The distribution folder is at: dist\usdjpy_assistant
    echo Please zip it manually.
)

echo.
pause
