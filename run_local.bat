@echo off
title StyleFinder Server
echo ==============================================================
echo  Starting StyleFinder Clothing Retrieval Service
echo  Network host: 0.0.0.0:8000 (accessible on LAN and localhost)
echo ==============================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python cir_server.py --host 0.0.0.0 --port 8000
pause

