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
if [ -f "$NPU_MODEL" ]; then
    echo "Quantized NPU model found: $NPU_MODEL"
else
    echo "Quantized NPU model not found."
    echo "Building modernbert_clamped_qdq.onnx directly on device (takes ~2 minutes)..."
    python "$ROOT_DIR/npu/build_clamped_22l.py"
fi

echo "=== Models Ready! ==="
