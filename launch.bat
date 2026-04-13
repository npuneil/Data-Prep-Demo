@echo off
:: ═══════════════════════════════════════════════════════
::  Data Prep Assistant — Quick Launcher
::  Double-click to start the demo
:: ═══════════════════════════════════════════════════════

title Data Prep Assistant
cd /d "%~dp0"

echo.
echo  =========================================
echo   * Data Prep Assistant
echo     On-Device AI - CoPilot+ PC Demo
echo  =========================================
echo.

:: Try PowerShell launcher first
powershell -ExecutionPolicy Bypass -File "%~dp0launch.ps1"

if %ERRORLEVEL% neq 0 (
    echo.
    echo  Trying direct Python launch...
    echo.
    
    if exist ".venv\Scripts\python.exe" (
        .venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
    ) else (
        python -m streamlit run app.py --server.port 8501
    )
)

pause
