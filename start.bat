@echo off
cd /d "%~dp0backend"

echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed. Make sure Python and pip are installed.
    pause
    exit /b 1
)

if not exist "model.pkl" (
    echo Model not found — training first ^(one-time, ~1 min^)...
    python train_model.py
    if errorlevel 1 (
        echo Training failed.
        pause
        exit /b 1
    )
    echo.
)

echo Starting Spam Detection Application...
echo Keep this window open while using the app.
echo.
python app.py
pause
