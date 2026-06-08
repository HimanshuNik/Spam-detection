@echo off
cd /d "%~dp0backend"
echo Starting Spam Detection API on http://127.0.0.1:5000 ...
echo Keep this window open while using Go Live or the browser app.
echo.
python app.py
pause
