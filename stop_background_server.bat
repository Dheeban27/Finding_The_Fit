@echo off
title Stop StyleFinder Background Server
echo Stopping StyleFinder and Cloudflare Tunnel...
taskkill /f /im cloudflared.exe >nul 2>&1
for /f "tokens=2" %%i in ('wmic process where "commandline like '%%cir_server.py%%' and name='python.exe'" get processid ^| findstr /r "[0-9]"') do (
    taskkill /f /pid %%i >nul 2>&1
)
echo StyleFinder background processes stopped.
pause

