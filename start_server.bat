@echo off
title InsightForge AI Server
echo.
echo  ================================
echo   InsightForge AI - Starting...  
echo  ================================
echo.
echo  Open in your browser:
echo  http://localhost:8000
echo.
cd /d "%~dp0"
"%~dp0venv\Scripts\python.exe" -m uvicorn api:app --host 0.0.0.0 --port 8000
pause
