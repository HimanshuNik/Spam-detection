#!/bin/bash
set -e

cd "$(dirname "$0")/backend"

# Ensure venv exists and activate it
if [ ! -d "../.venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv ../.venv
fi

# Activate virtual environment
source ../.venv/bin/activate || source ../.venv/Scripts/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Train model if needed
if [ ! -f "model.pkl" ]; then
    echo "Model not found — training first (one-time, ~1 min)..."
    python3 train_model.py
    if [ $? -ne 0 ]; then
        echo "Training failed."
        exit 1
    fi
fi

echo "Starting Spam Detection..."
python3 app.py
