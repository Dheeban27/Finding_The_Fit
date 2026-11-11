@echo off
title StyleFinder Public Tunnel
echo ==============================================================
echo  StyleFinder - Instant Public HTTPS URL (Cloudflare Tunnel)
echo  Routing to http://127.0.0.1:8000
echo ==============================================================
cd /d "%~dp0"
cloudflared.exe tunnel --protocol http2 --url http://127.0.0.1:8000
pause
