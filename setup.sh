#!/usr/bin/env bash
set -e

echo "=== Setting up Laya on Qualcomm Hexagon NPU (Rubik Pi 3) ==="

# 1. Check FastRPC CDSP device
if [ ! -e "/dev/fastrpc-cdsp" ]; then
    echo "Error: /dev/fastrpc-cdsp device not found. Ensure Qualcomm DSP drivers are loaded."
    exit 1
fi

# 2. Check or start cdsprpcd daemon
if ! pgrep -x "cdsprpcd" > /dev/null; then
    echo "Starting FastRPC daemon cdsprpcd..."
    if command -v cdsprpcd > /dev/null; then
        sudo cdsprpcd &
    else
        echo "Warning: cdsprpcd binary not found in PATH. Ensure qualcomm fastrpc package is installed."
    fi
fi

# 3. Create Python virtual environment if not present
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# 4. Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Setup complete! ==="
echo "To convert or download models, run: ./download_models.sh"
echo "To start the service: uvicorn app:app --host 0.0.0.0 --port 8000"
