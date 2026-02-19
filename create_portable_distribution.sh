#!/usr/bin/env bash
# ============================================================
#  USDJPY Assistant - Create Portable Distribution (macOS/Linux)
#  Builds the frontend, then creates a Windows self-contained zip.
#  Recipients on Windows unzip and double-click LAUNCH.bat.
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "============================================"
echo "  Creating Portable Distribution"
echo "============================================"
echo ""

# ============================================================
#  Build frontend first
# ============================================================
echo "Building frontend with latest changes..."
echo ""

if command -v node >/dev/null 2>&1; then
  if [ ! -d "frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    (cd frontend && npm install)
  fi
  (cd frontend && npm run build)
  if [ ! -f "frontend/dist/index.html" ]; then
    echo "[ERROR] frontend/dist/index.html not found after build!"
    exit 1
  fi
  echo "Frontend built successfully. Latest UI is included."
  echo ""
else
  echo "[WARNING] Node.js not found. Frontend will not be rebuilt."
  echo "If you have an old frontend/dist, it will be used."
  echo ""
  if [ ! -f "frontend/dist/index.html" ]; then
    echo "[ERROR] No frontend/dist found. Install Node.js and run this script again."
    exit 1
  fi
fi

# ============================================================
#  Clean and create output folder
# ============================================================
echo "Cleaning previous build..."
rm -rf dist/usdjpy_assistant
rm -f dist/python_embed.zip
mkdir -p dist/usdjpy_assistant

# ============================================================
#  Download Windows Embedded Python
# ============================================================
echo ""
echo "Downloading Python 3.12 embeddable (Windows)..."

PYTHON_URL="https://www.python.org/ftp/python/3.12.7/python-3.12.7-embed-amd64.zip"
GETPIP_URL="https://bootstrap.pypa.io/get-pip.py"

if command -v curl >/dev/null 2>&1; then
  curl -sL -o dist/python_embed.zip "$PYTHON_URL"
else
  echo "[ERROR] curl not found. Please install curl."
  exit 1
fi

if [ ! -f dist/python_embed.zip ]; then
  echo "[ERROR] Failed to download Python embeddable! Check your internet connection."
  exit 1
fi

echo "Extracting Python..."
unzip -q -o dist/python_embed.zip -d dist/usdjpy_assistant/python_embed
rm -f dist/python_embed.zip

# ============================================================
#  Configure Embedded Python for pip/site-packages
# ============================================================
echo "Configuring Python for pip support..."

PTH_FILE="dist/usdjpy_assistant/python_embed/python312._pth"
printf '%s\n' "python312.zip" "." "Lib\\site-packages" "import site" > "$PTH_FILE"
mkdir -p dist/usdjpy_assistant/python_embed/Lib/site-packages

# ============================================================
#  Download get-pip.py
# ============================================================
echo "Downloading get-pip.py..."
curl -sL -o dist/usdjpy_assistant/get-pip.py "$GETPIP_URL"
if [ ! -f dist/usdjpy_assistant/get-pip.py ]; then
  echo "[ERROR] Failed to download get-pip.py!"
  exit 1
fi

# ============================================================
#  Copy Application Files (exclude __pycache__, .pyc, etc.)
# ============================================================
echo ""
echo "Copying application files..."

RSYNC_EXCLUDE=(--exclude='__pycache__' --exclude='*.pyc' --exclude='.venv' --exclude='node_modules' --exclude='.git' --exclude='.DS_Store')

for dir in adapters api core storage; do
  if [ -d "$dir" ]; then
    rsync -a "${RSYNC_EXCLUDE[@]}" "$dir/" "dist/usdjpy_assistant/$dir/"
  fi
done

if [ -d "frontend/dist" ]; then
  mkdir -p dist/usdjpy_assistant/frontend
  rsync -a frontend/dist/ dist/usdjpy_assistant/frontend/dist/
fi

if [ -d "profiles/v1" ]; then
  rsync -a profiles/v1/ dist/usdjpy_assistant/profiles/
else
  mkdir -p dist/usdjpy_assistant/profiles
  [ -d "profiles" ] && rsync -a profiles/ dist/usdjpy_assistant/profiles/
fi

for f in run_api.py run_loop.py doctor_mt5.py snapshot_log.py close_trade.py review_stats.py requirements.txt; do
  [ -f "$f" ] && cp "$f" dist/usdjpy_assistant/
done

if [ -d ".streamlit" ]; then
  rsync -a .streamlit/ dist/usdjpy_assistant/.streamlit/
fi

mkdir -p dist/usdjpy_assistant/logs

# ============================================================
#  Create LAUNCH.bat
# ============================================================
echo "Creating LAUNCH.bat..."

cat > dist/usdjpy_assistant/LAUNCH.bat << 'LAUNCH_EOF'
@echo off
REM ============================================================
REM  USDJPY Assistant - One-Click Launcher
REM  First run installs dependencies (2-3 minutes).
REM  Subsequent runs start immediately.
REM ============================================================

cd /d "%~dp0"

REM Check if dependencies are installed (fastapi as marker)
if not exist "python_embed\Lib\site-packages\fastapi" (
    echo.
    echo ============================================
    echo   First Run Setup
    echo ============================================
    echo.
    echo Installing dependencies... This takes 2-3 minutes.
    echo Please wait...
    echo.
    python_embed\python.exe get-pip.py --quiet
    python_embed\python.exe -m pip install -r requirements.txt --quiet
    echo.
    echo Setup complete!
    echo.
)

echo ============================================
echo   USDJPY Assistant - Starting...
echo ============================================
echo.
echo The web app will open in your browser.
echo Keep this window open while using the app.
echo Press Ctrl+C to stop.
echo.

python_embed\python.exe run_api.py

pause
LAUNCH_EOF

# ============================================================
#  Create README.txt
# ============================================================
echo "Creating README.txt..."

cat > dist/usdjpy_assistant/README.txt << 'README_EOF'
============================================================
  USDJPY Assistant - Quick Start Guide
============================================================

PREREQUISITES:
  - Windows 10 or 11
  - MetaTrader 5 installed and logged into a DEMO account
  - Enable "Algo Trading" in MT5 (Tools > Options > Expert Advisors)
  - Add USDJPY to Market Watch in MT5

HOW TO USE:
  1. Open MetaTrader 5 and log in to your demo account
  2. Double-click LAUNCH.bat
  3. First run: Wait 2-3 minutes for setup (one-time only)
  4. Your browser opens to http://127.0.0.1:8000
  5. Select your account and start trading!

TROUBLESHOOTING:
  - "MT5 not initialized": Make sure MT5 is open and logged in
  - Browser doesn't open: Go to http://127.0.0.1:8000 manually
  - Port in use: Close other apps using port 8000

Keep the command window open while using the app.
Press Ctrl+C in the command window to stop.

============================================================
README_EOF

# ============================================================
#  Create ZIP
# ============================================================
echo ""
echo "Creating zip file..."

rm -f dist/usdjpy_assistant_portable.zip
(cd dist && zip -r -q usdjpy_assistant_portable.zip usdjpy_assistant)

if [ -f dist/usdjpy_assistant_portable.zip ]; then
  echo ""
  echo "============================================"
  echo "  Portable Distribution Created!"
  echo "============================================"
  echo ""
  echo "Location: dist/usdjpy_assistant_portable.zip"
  echo ""
  echo "Send this zip to recipients. They just need to:"
  echo "  1. Unzip to any folder"
  echo "  2. Double-click LAUNCH.bat"
  echo ""
  echo "No Python install required!"
  echo ""
else
  echo ""
  echo "[WARNING] Could not create zip file."
  echo "The distribution folder is at: dist/usdjpy_assistant"
  echo "Zip it manually if needed."
  exit 1
fi
