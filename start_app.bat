@echo off
title StyleFinder - AI Fashion Discovery
cd /d "%~dp0"

echo ======================================================================
echo           StyleFinder - Composed Fashion Retrieval System
echo ======================================================================
echo.
echo  Permanent Public URL : https://stylefinder-dheeban.loca.lt
echo  Local Network URL    : http://localhost:8000
echo.
echo  Fetching your current tunnel password (Public IP)...
for /f "delims=" %%a in ('powershell -Command "(Invoke-RestMethod -Uri https://api.ipify.org).Trim()"') do set "MY_IP=%%a"
echo  Tunnel Password      : %MY_IP%
echo.
echo ======================================================================
echo.

:: 1. Start StyleFinder Server
echo [1/2] Starting StyleFinder AI Server on port 8000...
start /b "" ".venv\Scripts\python.exe" cir_server.py --host 0.0.0.0 --port 8000

:: 2. Wait 4 seconds for server initialization
timeout /t 4 /nobreak >nul

:: 3. Connect Localtunnel with reserved permanent subdomain
echo [2/2] Connecting permanent public URL (https://stylefinder-dheeban.loca.lt)...
npx -y localtunnel --port 8000 --subdomain stylefinder-dheeban
