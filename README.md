# Laya on Rubik Pi 3

Experimental acceleration of the multilingual [Laya decision agent](https://huggingface.co/convaiinnovations/laya) on the Qualcomm Hexagon HTP V68 in the Rubik Pi 3 (QCS6490).

**Actual NPU fidelity passes the declared held-out target:** **3.5326% decision mismatch** and **2.5021% mean total variation** on 1,840 decisions excluded from calibration and model selection. The complete 2,000-decision public split has 3.55% mismatch. All preprocessed inputs match original Laya, with zero CPU EP fallbacks. Validation of 1024-token long inputs and the service deployment is still in progress.

The goal is decision mismatch **≤5%** and mean total variation **≤5%**, with identical preprocessed inputs. See [the evaluation method](docs/fidelity-method.md) and [held-out hardware evidence](reports/qualification-2026-09-29/heldout.json). These aggregate thresholds do not mean every individual prediction has less than 5% probability error.

## What runs where

`laya_npu.py` loads the original Laya agent and replaces only its encoder. The original tokenizer, sequence construction, decision heads, temperatures, and response formatting remain in use. Token embeddings and heads run on the CPU; the quantized encoder runs on HTP with CPU EP fallback disabled. A QNN failure is an error.

The checkpoint uses `max_len=1024` and `head_max_len=256`. The runtime selects a static ONNX bucket large enough for the unchanged input and rejects sequences beyond the available buckets instead of silently truncating them. Current measured candidates have a 768-token bucket; a validated set covering the complete 1024-token API budget is still pending.

The current recipe uses 16-bit activations, per-channel 8-bit convolution weights, native GELU, separate GeGLU outlier channels, and separate CLS/rest residual paths. Masked padding embeddings are zeroed without changing input tokens. Dynamic attention MatMuls approximate their 16-bit right operand using two supported 8-bit terms. Finally, [zero-input HTP calibration](npu/CONV_OFFSET_CALIBRATION.md) corrects measured per-channel Conv offsets. CPU ONNX results are diagnostics; hardware results decide whether a candidate is acceptable.

## Reproduce an experimental build

Build and calibrate on an x86 machine or WSL2 using Python 3.12 and [the tested build dependencies](requirements-fidelity-build.txt). No ARM cross compiler is required for this ONNX export path. The general `requirements.txt` includes the target QNN plugin and is distinct from this build environment.

From the repository root:

```bash
python3.12 -m venv .venv-build
source .venv-build/bin/activate
python -m pip install -r requirements-fidelity-build.txt
python download.py
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('LocalLLaMA/typed-decisions', 'all/test-00000-of-00001.parquet', repo_type='dataset', revision='c76749ec58bd8c3d2ea706b31c333a9059c38f90', local_dir='.work/dataset')"
```

Build the uncorrected `refined32` recipe:

```bash
python npu/build_fidelity.py \
  --model models/multilingual \
  --parquet .work/dataset/all/test-00000-of-00001.parquet \
  --length 768 --samples 32 --threads 8 \
  --activation-bits 16 --group-outliers --conv-linear --split-cls \
  --zero-pad-embeddings --refine-matmul-rhs all \
  --indices 0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375 \
  --output-dir .work/refined32
```

This creates FP32 and QDQ graphs, build metadata, and `manifest.json`. **The uncorrected graph fails hardware fidelity.** Follow the [Conv calibration workflow](npu/CONV_OFFSET_CALIBRATION.md) on the Pi to produce a separate corrected graph and manifest before evaluation. Keep each recipe in its own output directory and retain its metadata. Large checkpoints, generated graphs, and context binaries are not stored in Git.

For larger buckets on memory-limited WSL, `--low-memory-calibration` disables the calibration arena and merges MinMax ranges after every sample; it does not discard samples. `--append-long-calibration` adds full-length repeated-state variants from the same selected calibration cases while retaining the originals; with `--samples 32` this means 64 sequences. It is restricted to the original maximum-length bucket. Rebuild and calibrate each bucket independently. Combine compatible corrected bucket manifests with `python -m npu.merge_manifests --output npu/fidelity/manifest.json <768-manifest> <1024-manifest>`.

## Evaluate on the Pi

Copy the candidate directory, checkpoint, and pinned dataset to the Pi. Use the installed ARM QNN environment (currently ONNX Runtime 1.30.0 and `onnxruntime-qnn` 2.5.0). Before **every** QNN command, source `npu/env.sh`; it selects the wheel's matching backend, stub, and DSP skeleton. Mixing these with another vendor SDK caused device failures during investigation.

```bash
source .venv/bin/activate
source npu/env.sh
export LAYA_NPU_MANIFEST="$PWD/.work/corrected_768/manifest.json"
export LAYA_NPU_CONTEXT_CACHE=1
```

Capture unchanged Laya on the development selection, then compare the actual HTP candidate. Use new output paths for each experiment:

```bash
DEV=1,26,51,76,101,126,151,176,201,226,251,276,301,326,351,376
DATA=.work/dataset/all/test-00000-of-00001.parquet
python benchmark_fidelity.py --backend cpu --dataset-path "$DATA" \
  --indices "$DEV" --threads 4 --output .work/reference-dev.jsonl
python benchmark_fidelity.py --backend npu --dataset-path "$DATA" \
  --indices "$DEV" --threads 4 --reference .work/reference-dev.jsonl \
  --output .work/corrected-dev.json
```

The benchmark records checkpoint/graph hashes, input token and marker identities, probability errors, decision mismatches, and runtime statistics. A failed threshold returns exit code 1. A development pass still requires the declared held-out evaluation described in [the method](docs/fidelity-method.md). Supplementary Chinese, English, and 1024-token probes are available through `benchmark_probes.py`; capture their unchanged CPU reference before NPU comparison.

## Context cache and service status

Persistent QNN context generation and strict reload have been validated on the Pi: the corrected 768 graph compiled in 419.5 seconds and reloaded in 1.62 seconds. Its embedded context contains a single QNN EPContext node. Cache publication is atomic; keys bind graph SHA256, runtime versions, backend/stub/skeleton hashes, and provider settings. A cache schema or binary change requires fresh preparation.

To test context precompilation after loading the QNN environment:

```bash
export LAYA_NPU_CONTEXT_CACHE=1
python npu/compile_contexts.py --manifest "$LAYA_NPU_MANIFEST" \
  --model-dir models/multilingual --verify-reload
```

The CLI hashes the checkpoint without loading its tensors and compiles using only encoder configuration, reducing peak memory relative to loading the full agent first. `LAYA_NPU_CACHE_DIR` overrides the default `npu/fidelity/context_cache` directory. Compilation and cache reload are reported separately.

The FastAPI entry point remains `app:app`, with `/health` and `/predict`. Service files are experimental integration scaffolding; no candidate is currently presented as a qualified replacement for original Laya. Final service validation also requires buckets covering the original input budget and measured memory use under the service limit.

## Legacy results

The older 64-token UINT8/clamped implementation changed the sequence and head budgets and did not preserve original-model fidelity. Its historical full public-split agreement was 42.3% (57.7% mismatch). Four semantic probes did not establish general fidelity, and historical speed comparisons do not establish speedup for the corrected input contract.

`npu/build_clamped_22l.py`, `download_models.sh`, and [npu/REPORT.md](npu/REPORT.md) are legacy artifacts. Their clamp, latency, and accuracy statements are not validation of the current work. Use `npu/build_fidelity.py` and `benchmark_fidelity.py` for new experiments.

Author: [Eric Yu](https://github.com/EricYu123456). License: [MIT](LICENSE).
