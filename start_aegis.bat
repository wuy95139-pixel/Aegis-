@echo off
title Aegis
echo ============================================================
echo   Aegis - Multi-Agent Personal Assistant
echo ============================================================
echo.

:: Check for virtual environment
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

:: Check dependencies
echo Checking dependencies...
pip install -r requirements.txt --quiet 2>nul

echo.
echo Starting Aegis web server...
echo.
echo   Open http://127.0.0.1:7860 in your browser
echo   Press Ctrl+C to stop
echo.

:: Open browser after short delay
start "" http://127.0.0.1:7860

:: Start server
python -m uvicorn src.api.server:app --host 127.0.0.1 --port 7860 --reload --reload-dir src

pause
