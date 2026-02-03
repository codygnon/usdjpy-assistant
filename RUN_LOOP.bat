@echo off
setlocal enabledelayedexpansion

REM Run the 1-minute polling loop. The UI controls mode/kill-switch via logs/<profile>/runtime_state.json.

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

echo.
echo Available profiles:
for %%f in ("%~dp0profiles\v1\*.json") do (
  echo   %%~nxf
)
echo.
set /p PROFILE_FILE=Enter profile filename (e.g. cody_demo.json): 

set "PROFILE_PATH=%~dp0profiles\v1\%PROFILE_FILE%"
if not exist "%PROFILE_PATH%" (
  echo Profile not found: %PROFILE_PATH%
  pause
  exit /b 1
)

echo Using profile: %PROFILE_PATH%
echo Starting loop... (Ctrl+C to stop)

%PY% "%~dp0run_loop.py" --profile "%PROFILE_PATH%"
pause
