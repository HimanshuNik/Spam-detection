@echo off
cd /d "%~dp0backend"

if not exist "model.pkl" (
    echo Model not found — training first ^(one-time, ~1 min^)...
    python train_model.py
    if errorlevel 1 (
        echo Training failed. Check Python and run: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo.
)

echo Starting Spam Detection at http://127.0.0.1:5000
echo Open that URL in your browser — works from any code editor.
echo Keep this window open while using the app.
echo.
python app.py
pause
