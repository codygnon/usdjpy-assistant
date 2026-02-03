@echo off
REM ============================================================
REM  USDJPY Assistant - Create Portable Distribution
REM  Builds the frontend with ALL latest changes, then creates
REM  a self-contained zip. Recipients unzip and double-click LAUNCH.bat.
REM ============================================================

setlocal enabledelayedexpansion

echo.
echo ============================================
echo   Creating Portable Distribution
echo ============================================
echo.

cd /d "%~dp0"

REM ============================================================
REM  Build frontend first (so zip includes all latest UI changes)
REM ============================================================
echo Building frontend with latest changes...
echo.

where node >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Node.js not found. Frontend will not be rebuilt.
    echo If you have an old frontend\dist, it will be used.
    echo Install Node.js and run this script again for latest UI.
    echo.
    choice /c YN /m "Continue with existing frontend\dist?"
    if errorlevel 2 exit /b 0
) else (
    cd frontend
    if not exist "node_modules" (
        echo Installing frontend dependencies...
        call npm install
        if %ERRORLEVEL% neq 0 (
            echo [ERROR] npm install failed!
            cd ..
            pause
            exit /b 1
        )
    )
    echo Running npm run build...
    call npm run build
    cd ..
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Frontend build failed!
        pause
        exit /b 1
    )
    if not exist "frontend\dist\index.html" (
        echo [ERROR] frontend\dist\index.html not found after build!
        pause
        exit /b 1
    )
    echo Frontend built successfully. Latest UI is included.
    echo.
)

REM Clean and create output folder
echo Cleaning previous build...
if exist "dist\usdjpy_assistant" rmdir /s /q "dist\usdjpy_assistant"
if exist "dist\python_embed.zip" del /q "dist\python_embed.zip"
mkdir "dist\usdjpy_assistant"

REM ============================================================
REM  Download Embedded Python
REM ============================================================
echo.
echo Downloading Python 3.12 embeddable...

set PYTHON_URL=https://www.python.org/ftp/python/3.12.7/python-3.12.7-embed-amd64.zip
set PYTHON_ZIP=dist\python_embed.zip

powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_ZIP%' -UseBasicParsing}"

if not exist "%PYTHON_ZIP%" (
    echo [ERROR] Failed to download Python embeddable!
    echo Please check your internet connection.
    pause
    exit /b 1
)

echo Extracting Python...
powershell -Command "Expand-Archive -Path '%PYTHON_ZIP%' -DestinationPath 'dist\usdjpy_assistant\python_embed' -Force"

REM Clean up zip
del /q "%PYTHON_ZIP%"

REM ============================================================
REM  Configure Embedded Python for pip/site-packages
REM ============================================================
echo Configuring Python for pip support...

set PTH_FILE=dist\usdjpy_assistant\python_embed\python312._pth

REM Rewrite the _pth file to enable site-packages
(
    echo python312.zip
    echo .
    echo Lib\site-packages
    echo import site
) > "%PTH_FILE%"

REM Create Lib\site-packages folder
mkdir "dist\usdjpy_assistant\python_embed\Lib\site-packages" 2>nul

REM ============================================================
REM  Download get-pip.py
REM ============================================================
echo Downloading get-pip.py...

set GETPIP_URL=https://bootstrap.pypa.io/get-pip.py
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%GETPIP_URL%' -OutFile 'dist\usdjpy_assistant\get-pip.py' -UseBasicParsing}"

if not exist "dist\usdjpy_assistant\get-pip.py" (
    echo [ERROR] Failed to download get-pip.py!
    pause
    exit /b 1
)

REM ============================================================
REM  Copy Application Files
REM ============================================================
echo.
echo Copying application files...

REM Create exclude file for xcopy (to skip __pycache__, .pyc, etc.)
(
    echo __pycache__
    echo .pyc
    echo .venv
    echo node_modules
    echo .git
    echo .DS_Store
) > "dist\xcopy_exclude.txt"

REM Copy Python packages (excluding __pycache__)
xcopy /s /e /i /exclude:dist\xcopy_exclude.txt "adapters" "dist\usdjpy_assistant\adapters"
xcopy /s /e /i /exclude:dist\xcopy_exclude.txt "api" "dist\usdjpy_assistant\api"
xcopy /s /e /i /exclude:dist\xcopy_exclude.txt "core" "dist\usdjpy_assistant\core"
xcopy /s /e /i /exclude:dist\xcopy_exclude.txt "storage" "dist\usdjpy_assistant\storage"

REM Copy built frontend
if exist "frontend\dist" (
    xcopy /s /e /i "frontend\dist" "dist\usdjpy_assistant\frontend\dist"
)

REM Copy profiles (only the v1 folder for distribution)
if exist "profiles\v1" (
    xcopy /s /e /i "profiles\v1" "dist\usdjpy_assistant\profiles"
) else (
    xcopy /s /e /i "profiles" "dist\usdjpy_assistant\profiles"
)

REM Copy main scripts
copy "run_api.py" "dist\usdjpy_assistant\"
copy "run_loop.py" "dist\usdjpy_assistant\"
copy "doctor_mt5.py" "dist\usdjpy_assistant\"
copy "snapshot_log.py" "dist\usdjpy_assistant\"
copy "close_trade.py" "dist\usdjpy_assistant\"
copy "review_stats.py" "dist\usdjpy_assistant\"

REM Copy requirements
copy "requirements.txt" "dist\usdjpy_assistant\"

REM Copy streamlit config if exists
if exist ".streamlit" (
    xcopy /s /e /i ".streamlit" "dist\usdjpy_assistant\.streamlit"
)

REM Create empty logs folder
mkdir "dist\usdjpy_assistant\logs" 2>nul

REM Clean up exclude file
del /q "dist\xcopy_exclude.txt"

REM ============================================================
REM  Create LAUNCH.bat
REM ============================================================
echo Creating LAUNCH.bat...

(
    echo @echo off
    echo REM ============================================================
    echo REM  USDJPY Assistant - One-Click Launcher
    echo REM  First run installs dependencies ^(2-3 minutes^).
    echo REM  Subsequent runs start immediately.
    echo REM ============================================================
    echo.
    echo cd /d "%%~dp0"
    echo.
    echo REM Check if dependencies are installed ^(fastapi as marker^)
    echo if not exist "python_embed\Lib\site-packages\fastapi" ^(
    echo     echo.
    echo     echo ============================================
    echo     echo   First Run Setup
    echo     echo ============================================
    echo     echo.
    echo     echo Installing dependencies... This takes 2-3 minutes.
    echo     echo Please wait...
    echo     echo.
    echo     python_embed\python.exe get-pip.py --quiet
    echo     python_embed\python.exe -m pip install -r requirements.txt --quiet
    echo     echo.
    echo     echo Setup complete!
    echo     echo.
    echo ^)
    echo.
    echo echo ============================================
    echo echo   USDJPY Assistant - Starting...
    echo echo ============================================
    echo echo.
    echo echo The web app will open in your browser.
    echo echo Keep this window open while using the app.
    echo echo Press Ctrl+C to stop.
    echo echo.
    echo.
    echo python_embed\python.exe run_api.py
    echo.
    echo pause
) > "dist\usdjpy_assistant\LAUNCH.bat"

REM ============================================================
REM  Create README.txt
REM ============================================================
echo Creating README.txt...

(
    echo ============================================================
    echo   USDJPY Assistant - Quick Start Guide
    echo ============================================================
    echo.
    echo PREREQUISITES:
    echo   - Windows 10 or 11
    echo   - MetaTrader 5 installed and logged into a DEMO account
    echo   - Enable "Algo Trading" in MT5 ^(Tools ^> Options ^> Expert Advisors^)
    echo   - Add USDJPY to Market Watch in MT5
    echo.
    echo HOW TO USE:
    echo   1. Open MetaTrader 5 and log in to your demo account
    echo   2. Double-click LAUNCH.bat
    echo   3. First run: Wait 2-3 minutes for setup ^(one-time only^)
    echo   4. Your browser opens to http://127.0.0.1:8000
    echo   5. Select your account and start trading!
    echo.
    echo TROUBLESHOOTING:
    echo   - "MT5 not initialized": Make sure MT5 is open and logged in
    echo   - Browser doesn't open: Go to http://127.0.0.1:8000 manually
    echo   - Port in use: Close other apps using port 8000
    echo.
    echo Keep the command window open while using the app.
    echo Press Ctrl+C in the command window to stop.
    echo.
    echo ============================================================
) > "dist\usdjpy_assistant\README.txt"

REM ============================================================
REM  Create ZIP
REM ============================================================
echo.
echo Creating zip file...

if exist "dist\usdjpy_assistant_portable.zip" del /q "dist\usdjpy_assistant_portable.zip"

powershell -Command "Compress-Archive -Path 'dist\usdjpy_assistant' -DestinationPath 'dist\usdjpy_assistant_portable.zip' -Force"

if %ERRORLEVEL% equ 0 (
    echo.
    echo ============================================
    echo   Portable Distribution Created!
    echo ============================================
    echo.
    echo Location: dist\usdjpy_assistant_portable.zip
    echo.
    echo Send this zip to recipients. They just need to:
    echo   1. Unzip to any folder
    echo   2. Double-click LAUNCH.bat
    echo.
    echo No Python install required!
    echo.
) else (
    echo.
    echo [WARNING] Could not create zip file automatically.
    echo The distribution folder is at: dist\usdjpy_assistant
    echo Please zip it manually.
)

echo.
pause
