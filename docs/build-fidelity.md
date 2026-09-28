# Build and validate the NPU models

Build and calibrate on an x86 machine or WSL2 using Python 3.12 and [the tested build dependencies](../requirements-fidelity-build.txt). No ARM cross compiler is required for this ONNX export path. The general `requirements.txt` includes the target QNN plugin and is distinct from this build environment.

From the repository root:

```bash
python3.12 -m venv .venv-build
source .venv-build/bin/activate
python -m pip install -r requirements-fidelity-build.txt
HF_HUB_OFFLINE=0 python download.py
HF_HUB_OFFLINE=0 python -c "from huggingface_hub import hf_hub_download; hf_hub_download('LocalLLaMA/typed-decisions', 'all/test-00000-of-00001.parquet', repo_type='dataset', revision='c76749ec58bd8c3d2ea706b31c333a9059c38f90', local_dir='.work/dataset')"
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

This creates FP32 and QDQ graphs, build metadata, and `manifest.json`. **The uncorrected graph fails hardware fidelity.** Follow the [Conv calibration workflow](../npu/CONV_OFFSET_CALIBRATION.md) on the Pi to produce a separate corrected graph and manifest before evaluation. Keep each recipe in its own output directory and retain its metadata. Large checkpoints, generated graphs, and context binaries are not stored in Git.

For larger buckets on memory-limited WSL, `--low-memory-calibration` disables the calibration arena and merges MinMax ranges after every sample; it does not discard samples. `--append-long-calibration` adds full-length repeated-state variants from the same selected calibration cases while retaining the originals; with `--samples 32` this means 64 sequences. It is restricted to the original maximum-length bucket. Rebuild and calibrate each bucket independently. Combine compatible corrected bucket manifests with `python -m npu.merge_manifests --output npu/fidelity/manifest.json <768-manifest> <1024-manifest>`.

The separately selected 1024-token recipe also folds interior LayerNorm affine parameters into the adjacent projections. Its calibration uses the same reserved cases, including their full-length variants:

```bash
python npu/build_fidelity.py \
  --model models/multilingual \
  --parquet .work/dataset/all/test-00000-of-00001.parquet \
  --length 1024 --samples 32 --threads 8 \
  --activation-bits 16 --group-outliers --conv-linear --split-cls \
  --zero-pad-embeddings --refine-matmul-rhs all --fold-norms \
  --append-long-calibration --low-memory-calibration \
  --indices 0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375 \
  --output-dir .work/folded1024_long
```

The builder materializes folded unit LayerNorm parameters as static U8 gamma and I32 zero bias, preserving their exact dequantized values; V68 rejects the U16 gamma aliases produced by the exporter. Run the same Conv calibration workflow on this bucket's own graph. Do not reuse the 768-token correction. The [candidate declaration](../reports/evaluation-plans/long-input-candidate-static-gamma.json) records selection before the independent long-input NPU evaluation.

Use a fresh output directory for a new recipe. The builder rejects existing quantized graphs and incompatible manifests before writing. `--reuse-export` only finishes a verified FP32 export when no quantized graph exists; its sidecar must match the checkpoint, tokenizer/configuration files, calibration dataset, and requested export settings. Older sidecars without those input hashes require a fresh export.

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

The benchmark records checkpoint/graph hashes, input token and marker identities, probability errors, decision mismatches, and runtime statistics. A failed threshold returns exit code 1. A development pass still requires the declared held-out evaluation described in [the method](fidelity-method.md). Supplementary Chinese, English, and 1024-token probes are available through `benchmark_probes.py`; capture their unchanged CPU reference before NPU comparison.

## Assemble the deployment set

Correct and validate both buckets before activation. The 768 recipe above can
produce `.work/corrected_768`; repeat the calibration workflow for the 1024
source graph with separate probe/correction directories and a target such as
`.work/corrected_1024/backbone_1024_a16w8.onnx`. Keep the ONNX file, emitted
manifest, and adjacent `.offsets.json` together when copying to stable paths.

The measured Pi deployment uses this layout:

```text
npu/fidelity/
  manifest.json
  qualified-2026-09-29/
    768/   # corrected graph, per-bucket manifest and provenance
    1024/  # corrected graph, per-bucket manifest and provenance
  context_cache/
```

Do not overwrite an active manifest while assembling another candidate. Merge
compatible per-bucket manifests into a fresh `manifest.next.json`, prepare their
contexts on the Pi, and follow [deployment and rollback](deployment.md) to switch
the service. Context creation is a separate offline operation with higher memory
requirements than cached inference.

See [the evaluation method](fidelity-method.md) for the original input contract,
frozen dataset splits, and held-out validation. External adapted datasets have a
separate [reproduction guide](../reports/external-2026-09-29/README.md).
