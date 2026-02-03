@echo off
setlocal enabledelayedexpansion

REM Run the Streamlit UI from this folder.
REM Assumes: MT5 is open and logged in (demo recommended).

cd /d "%~dp0"

REM Prefer venv python if present
if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

echo Using Python: %PY%
echo Starting UI...

REM Streamlit must be installed in the chosen environment.
%PY% -m streamlit run "%~dp0ui_app.py"

