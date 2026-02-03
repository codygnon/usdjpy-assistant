@echo off
REM ============================================================
REM  USDJPY Assistant - Build Frontend
REM  This script builds the React frontend for production.
REM ============================================================

echo.
echo ============================================
echo   Building Frontend...
echo ============================================
echo.

cd /d "%~dp0\frontend"

REM Check if Node.js is installed
where node >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Node.js is not installed!
    echo.
    echo Please install Node.js from https://nodejs.org/
    echo.
    pause
    exit /b 1
)

REM Install dependencies if needed
if not exist "node_modules" (
    echo Installing dependencies...
    call npm install
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to install dependencies!
        pause
        exit /b 1
    )
)

REM Build the frontend
echo Building for production...
call npm run build

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Frontend built successfully!
echo   Output: frontend/dist/
echo ============================================
echo.
pause
