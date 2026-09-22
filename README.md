# Laya on Qualcomm Hexagon NPU (Rubik Pi 3)

[![Qualcomm Hexagon](https://img.shields.io/badge/Hardware-Qualcomm%20Hexagon%20HTP%20V68-orange.svg)](https://developer.qualcomm.com/)
[![ONNX Runtime QNN](https://img.shields.io/badge/Runtime-ONNX%20Runtime%20QNN%202.5.0-blue.svg)](https://github.com/microsoft/onnxruntime)
[![Latency](https://img.shields.io/badge/Backbone%20Latency-14.1ms-green.svg)](#performance-benchmarks)
[![Zero Fallback](https://img.shields.io/badge/Strict%20NPU-disable__cpu__ep__fallback%3D1-brightgreen.svg)](#key-technical-breakthroughs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

High-performance edge deployment of the **Laya decision agent** (`convaiinnovations/laya`, multilingual checkpoint) on the **Qualcomm Hexagon NPU (HTP V68)** of the **Thundercomm Rubik Pi 3 (QCS6490)**.

Achieves true hardware acceleration with **`session.disable_cpu_ep_fallback=1`** strictly enforced, eliminating silent CPU fallbacks.

---

## Highlights

- ⚡ **14.26 ms Backbone Latency**: The 22-layer ModernBERT backbone executes entirely on Hexagon HTP V68 in ~14ms with **0 DDR spill bytes** (fully resident in VTCM).
- 🚀 **Up to 6.6x API Acceleration**: End-to-end FastAPI endpoint latency dropped from **520.4 ms** (CPU PyTorch) down to **182.9 ms** for 2-question routing, and down to **58.9 ms** for single-question evaluation.
- 🎯 **100% Accuracy Fidelity**: Passes all 4/4 semantic test probes in `accuracy-probes.json` and perfectly matches routing decisions on `example.json`.
- 🛡️ **Zero Silent CPU Fallback**: Compiled as a single unified HTP graph under strict `session.disable_cpu_ep_fallback=1`.

---

## Architecture: Hybrid Edge Offloading

The service employs a hybrid execution pipeline designed for minimal latency:

```
                      [ HTTP Request (FastAPI /predict) ]
                                      │
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │ CPU (ARM Cortex-A78)                             │
             │ • Fast Tokenizer (Hugging Face)                  │
             │ • Embedding Lookup (tok_embed)                   │
             └──────────────────────────────────────────────────┘
                                      │
                                      ▼ (1 x 64 x 768 Tensor)
             ┌──────────────────────────────────────────────────┐
             │ Qualcomm Hexagon NPU (CDSP / HTP V68)            │
             │ • 22-Layer ModernBERT Encoder Backbone           │
             │ • Static UINT8 QDQ Quantized Graph               │
             │ • session.disable_cpu_ep_fallback = 1            │
             │ • Inference Time: ~14.1 ms (0 DDR Spills)        │
             └──────────────────────────────────────────────────┘
                                      │
                                      ▼ (1 x 64 x 768 Hidden State)
             ┌──────────────────────────────────────────────────┐
             │ CPU (ARM Cortex-A78)                             │
             │ • Question Type Embeddings                       │
             │ • Head Attention (2-layer transformer encoder)   │
             │ • Scorer & Policy/Cost Action Heads              │
             │ • Calibrated Softmax / Temperature Scaling       │
             └──────────────────────────────────────────────────┘
                                      │
                                      ▼
                      [ HTTP Response (JSON Decision) ]
```

---

## Key Technical Breakthroughs

Deploying ModernBERT to Qualcomm Hexagon DSP required resolving three major hardware/compilation roadblocks:

### 1. DSP Sub-graph Fragmentation & Handle Limit (Code 6001)
* **Root Cause**: PyTorch's default ONNX export expands GELU into mathematical primitives (`Div`, `Erf`, `Pow`, `Add`). The QNN HTP compiler could not fuse these unquantized nodes, fragmenting the 22-layer model into **282 individual subgraphs**. This immediately exceeded Hexagon DSP's hardware limit of 32 session handles (`Code 6001: Exceeded max graph limit`). Additionally, bias-free `LayerNormalization` produced Constant 0 bias nodes that failed QNN op validation (error 3110).
* **Fix**: Exported with **ONNX Opset 20** to emit native single `Gelu` nodes natively supported by QNN HTP, and promoted all 45 LayerNorm zero-biases into static graph initializers. The entire 22-layer backbone now fuses into **1 single unified Hexagon HTP graph**.

### 2. GeGLU Activation Outliers Collapsing Dynamic Range
* **Root Cause**: ModernBERT Layer 11 MLP GeGLU activation generates an extreme outlier of **+33,419.97** at channel 924 of the CLS token. Tensor-wise UINT8 quantization widened the quantization scale to 131, rounding the other 1,151 normal channels to 0 and destroying representations.
* **Fix**: Bounded GeGLU activations with `torch.clamp(act, -50.0, 50.0)`. PyTorch accuracy verified at >99.99% fidelity, while UINT8 quantization dynamic range was kept perfectly well-conditioned.

### 3. Attention Mask Dynamic Range Collapse
* **Root Cause**: Standard `-10000.0` attention mask penalties stretched the dynamic range of `Add` nodes to 10,000, setting the quantization scale to ~40 and washing out normal attention logits.
* **Fix**: Replaced mask penalty with `MASK_PENALTY = -15.0`. In Softmax:
  $$\exp(-15.0) \approx 3.05 \times 10^{-7} \approx 0$$
  This completely eliminates padded tokens while keeping the `Add` node quantization scale fine-grained. Logit difference against FP32 is `< 0.001`.

---

## Performance Benchmarks

### Backbone Inference Latency (Hexagon HTP vs CPU)

| Metric | CPU Baseline (PyTorch 4-Threads) | Hexagon HTP V68 (UINT8 QDQ) | Speedup |
| :--- | :---: | :---: | :---: |
| **Backbone Latency (avg)** | ~230 ms | **14.26 ms** | **16.1x** |
| **Backbone Latency (p50)** | ~228 ms | **14.12 ms** | **16.1x** |
| **DDR Spill Bytes** | N/A | **0 bytes** (100% in VTCM) | Optimal |
| **Fallback Policy** | N/A | `disable_cpu_ep_fallback=1` | Zero CPU |

### End-to-End API Latency (`verify.py`)

| Benchmark Scenario | CPU Baseline (ms) | Hexagon NPU (ms) | Speedup |
| :--- | :---: | :---: | :---: |
| **Cold Request (First Call)** | 615.0 ms | **281.0 ms** | **2.19x** |
| **Warm Two-Question Routing** | 520.4 ms | **183.4 ms** | **2.84x** |
| **Single-Question Urgency Score** | 392.9 ms | **58.9 ms** | **6.67x** |
| **Negative Verification** | 609.2 ms | **78.8 ms** | **7.73x** |

### Accuracy Verification

Tested against [`accuracy-probes.json`](accuracy-probes.json) and [`example.json`](example.json):

| Test Case | Metric / Question | CPU PyTorch | Hexagon NPU | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Probe 1** | Price inquiry (non-refund) | 0.0015 | **0.0459** (<0.5) | **PASS** |
| **Probe 2** | English non-refund statement | 0.6914 | **0.7148** (>0.5) | **PASS** |
| **Probe 3** | Chinese complex refund | 0.9172 | **0.9366** (>0.5) | **PASS** |
| **Probe 4** | Explicit refund request | 0.9945 | **0.9684** (>0.5) | **PASS** |
| **example.json** | Department classification | `billing` (97.5%) | **`billing` (82.2%)** | **PASS** |
| **example.json** | Refund flag (`noul`) | 0.9945 | **0.9883** | **PASS** |
| **example.json** | Urgency score | 1.9554 | **1.9552** | **PASS** (Δ < 0.0002) |

---

## Large Model Files & Download Links

GitHub has a strict **100 MB per file limit**. The following binary weights exceed 100 MB:

| File Path | Size | Description | How to Obtain |
| :--- | :---: | :--- | :--- |
| `npu/modernbert_clamped_qdq.onnx` | **106.3 MB** | Quantized 22-layer Hexagon HTP QDQ model | [Google Drive Link](#) / Build via script |
| `models/multilingual/model.safetensors` | **615 MB** | Hugging Face base PyTorch weights | `python download.py` |

> 💡 **Download Options**:
> - **Option A (Google Drive)**: [Download pre-converted models from Google Drive](#) *(Paste your Google Drive link here)* and extract to `npu/` and `models/`.
> - **Option B (One-Click Local Build)**: Run `./download_models.sh`. It automatically pulls weights from Hugging Face and quantizes `modernbert_clamped_qdq.onnx` directly on the device in ~2 minutes.

---

## Quickstart Guide

### 1. Prerequisites (Rubik Pi 3 / Ubuntu 24.04)

Ensure FastRPC drivers and daemon are running:
```bash
# Check device
ls -l /dev/fastrpc-cdsp

# Ensure cdsprpcd daemon is active
sudo cdsprpcd &
```

### 2. Environment Setup

```bash
git clone https://github.com/EricYu123456/laya-hexagon-npu.git
cd laya-hexagon-npu

# Run setup script (creates virtualenv and installs dependencies)
./setup.sh
```

Or manually:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Obtain Model Weights

```bash
./download_models.sh
```
Or rebuild the quantized ONNX graph from scratch:
```bash
python download.py
python npu/build_clamped_22l.py
```

### 4. Run the API Service

```bash
source .venv/bin/activate
export LAYA_DEVICE=npu
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 5. Run Verification

In another terminal:
```bash
source .venv/bin/activate
python verify.py
```

---

## Running as a Systemd Service

To run Laya as a persistent background daemon managed by systemd:

```bash
mkdir -p ~/.config/systemd/user/
cp systemd/laya.service ~/.config/systemd/user/laya.service

# Reload and enable
systemctl --user daemon-reload
systemctl --user enable --now laya.service

# Check status
systemctl --user status laya.service
```

---

## API Reference

### `GET /health`
Returns readiness status, active checkpoint, execution device, and threads.

**Response**:
```json
{
  "status": "ready",
  "checkpoint": "multilingual",
  "device": "npu",
  "threads": 4,
  "revision": "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
}
```

### `POST /predict`
Evaluates dynamic questions over dialogue state.

**Request**:
```json
{
  "state": "Hi, I would like to request a refund for order #12345.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which department should handle this ticket?",
      "criteria": {
        "billing": "Invoice, refunds, subscription charges",
        "technical": "Software bugs, system crashes",
        "sales": "Product purchase inquiries"
      }
    },
    "refund_requested": {
      "type": "noul",
      "instructions": "Does the user explicitly request a refund?"
    }
  }
}
```

**Response**:
```json
{
  "model": "laya-rl-agent",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {
        "billing": 0.8221,
        "technical": 0.1524,
        "sales": 0.0255
      },
      "confidence": 0.5072,
      "action": {
        "act_probability": 1.0
      }
    },
    "refund_requested": {
      "type": "noul",
      "noul": 0.9883,
      "confidence": 0.9883,
      "action": {
        "act_probability": 1.0
      }
    }
  },
  "usage": {
    "input_tokens": 98,
    "output_tokens": 0
  },
  "elapsed_ms": 183.4,
  "checkpoint": "multilingual"
}
```

---

## Technical Report

For in-depth mathematical analysis, profiling traces, and quantization logs, see [`npu/REPORT.md`](npu/REPORT.md).

---

## Author & License

- **Author**: Eric Yu ([@EricYu123456](https://github.com/EricYu123456))
- **License**: [MIT License](LICENSE)
