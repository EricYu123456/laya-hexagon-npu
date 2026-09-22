#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT_DIR/models" "$ROOT_DIR/npu"

echo "=== Model Preparation for Laya Hexagon NPU ==="

# 1. Download PyTorch Base Model from Hugging Face if not present
if [ ! -f "$ROOT_DIR/models/multilingual/model.safetensors" ]; then
    echo "Downloading base PyTorch weights from Hugging Face (convaiinnovations/laya)..."
    python "$ROOT_DIR/download.py"
else
    echo "Base PyTorch model found at models/multilingual/model.safetensors."
fi

# 2. Check for Quantized NPU Model (modernbert_clamped_qdq.onnx)
NPU_MODEL="$ROOT_DIR/npu/modernbert_clamped_qdq.onnx"
RELEASE_URL="https://github.com/EricYu123456/laya-hexagon-npu/releases/download/v1.0.0/modernbert_clamped_qdq.onnx"

if [ -f "$NPU_MODEL" ]; then
    echo "Quantized NPU model found: $NPU_MODEL"
else
    echo "Quantized NPU model not found."
    echo "Attempting to download pre-converted weights from GitHub Releases ($RELEASE_URL)..."
    if curl -fL "$RELEASE_URL" -o "$NPU_MODEL"; then
        echo "Download successful: $NPU_MODEL"
    else
        echo "Download failed. Falling back to building modernbert_clamped_qdq.onnx directly on device..."
        python "$ROOT_DIR/npu/build_clamped_22l.py"
    fi
fi

echo "=== Models Ready! ==="
