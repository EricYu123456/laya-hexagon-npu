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

## Precision refinement recipe

The 2026-10-03 recipe passed all nine historical NPU suites and four declared
subsets. The highest decision mismatch rate was **0.6%** and the highest mean
total variation was **0.622212%**, with strict HTP execution and zero CPU encoder
fallbacks. See the [qualification evidence](../reports/qualification-2026-10-03/)
and [audit](../reports/qualification-2026-10-03/one-percent-audit.json). These are
aggregate historical regression limits, not a per-example error guarantee or a
new independent held-out evaluation.

The qualified recipe separates residual features into bands using only the
reserved calibration inputs. Ordinary features no longer share a quantization
range with a few very large residual channels. CLS and other tokens remain
separate within each band. Each token uses one common scale before LayerNorm;
the scaled epsilon approximation must pass the builder's `0.01` maximum FP32
export-error gate before quantization.

Build both buckets with the same feature thresholds and projection recipe:

```bash
CAL=0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375
for LENGTH in 768 1024; do
  EXTRA=()
  if [ "$LENGTH" = 1024 ]; then EXTRA+=(--append-long-calibration); fi
  python npu/build_fidelity.py \
    --model models/multilingual \
    --parquet .work/dataset/all/test-00000-of-00001.parquet \
    --length "$LENGTH" --samples 32 --threads 6 \
    --activation-bits 16 --group-outliers --conv-linear \
    --split-cls --split-features --feature-thresholds 100,500,2000 \
    --zero-pad-embeddings --refine-matmul-rhs all --fold-norms \
    --low-memory-calibration "${EXTRA[@]}" --indices "$CAL" \
    --output-dir ".work/features32_folded${LENGTH}"
  python -m npu.weight_refinement \
    --source ".work/features32_folded${LENGTH}/backbone_${LENGTH}_a16w8.onnx" \
    --fp32 ".work/features32_folded${LENGTH}/backbone_${LENGTH}_fp32.onnx" \
    --source-manifest ".work/features32_folded${LENGTH}/manifest.json" \
    --checkpoint models/multilingual/model.safetensors \
    --out-dir ".work/features32_weight16_${LENGTH}"
  python -m npu.affine_layernorm \
    --source ".work/features32_weight16_${LENGTH}/backbone_${LENGTH}_a16w8.onnx" \
    --fp32 ".work/features32_folded${LENGTH}/backbone_${LENGTH}_fp32.onnx" \
    --source-manifest ".work/features32_weight16_${LENGTH}/manifest.json" \
    --checkpoint models/multilingual/model.safetensors \
    --expected-layernorms 3 \
    --out-dir ".work/features32_precision16_${LENGTH}"
done
```

The recorded feature groups contain `631, 127, 6, 4` channels for this pinned
checkpoint and calibration selection. The group assignment, export checks and
input hashes are saved in the build metadata and FP32 sidecar. No evaluation
answers determine the groups. The 1024 bucket includes 32 original and 32
full-length calibration sequences; 768 uses the original 32 sequences.

The affine step refines the embedding norm and the CLS/rest final norms. It
verifies the weight graph, checkpoint and matching FP32 sidecar, then retains
the source build metadata, feature groups and weight provenance in its new
output directory. Existing outputs and corrected source graphs are rejected.

Next prepare and measure **fresh Conv offset probes for each bucket**, using
the graphs under `.work/features32_precision16_${LENGTH}` as the source.
The [weight-refinement guide](../npu/WEIGHT_REFINEMENT.md) explains
the two-term INT8 weights, source checks and hardware calibration. Retain the
original export, intermediate graphs and metadata so every transform can be
audited. CPU QDQ results are diagnostic; final acceptance requires
[all historical suites on the actual NPU](one-percent-qualification.md).

The [recorded 768-token development comparison](../reports/precision-development-2026-10-03/)
shows the complete recipe's accuracy/latency tradeoff. Two-term weights alone
did not improve mean TV on that selection; qualification applies to the complete
feature-band, weight, affine and Conv-offset recipe.

## Measured context compilation

Context preparation runs on the Pi after graph-specific Conv correction. The
following measurements include `compile_contexts.py --verify-reload`, using
the same QNN environment as qualification:

| Bucket | Command elapsed time | First context preparation | Cached reload | Peak process RSS |
|---|---:|---:|---:|---:|
| 768 | 21 min 25.39 s | 1191.82 s | 9.75 s | 6.5412 GiB |
| 1024 | 1 h 22 min 44 s | 4737.72 s | 12.59 s | 6.5901 GiB |

Both commands exited successfully and verified cache reload with zero CPU
encoder fallbacks. The 1024 build used two temporary 8 GiB swap files, 16 GiB
configured total; both were removed before qualification. Peak process swap
was not measured. These RSS values describe compilation, not steady service
memory. See the [768 compile evidence](../reports/precision-development-2026-10-03/feature768-compile-summary.json)
and [1024 build evidence](../reports/precision-build-2026-10-03/).

## Historical five-percent recipes

These commands reproduce the earlier `refined32` and folded-1024 candidates,
whose reports remain under `reports/qualification-2026-09-29`. Build the
uncorrected `refined32` recipe:

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

The qualified artifacts on the Pi use this layout:

```text
npu/fidelity/
  manifest.json
  manifest.one-percent-2026-10-03.json  # evaluated two-bucket manifest
  precision-2026-10-03/
    768/   # corrected graph, per-bucket manifest and provenance
    1024/  # corrected graph, per-bucket manifest and provenance
  qualified-2026-09-29/               # retained earlier model set
```

The evaluated manifest points to `precision-2026-10-03/{768,1024}`; its exact
contents are retained in the [qualification manifest](../reports/qualification-2026-10-03/candidate-manifest.json).
Qualification used the contexts in `.work/one-percent-context-cache`, whose
hashes were [recorded before the run](../reports/qualification-2026-10-03/context-cache-before-run.json).
The active service's `manifest.json` is managed separately during deployment.

Do not overwrite an active manifest while assembling another candidate. Merge
compatible per-bucket manifests into a fresh `manifest.next.json`, prepare their
contexts on the Pi, and follow [deployment and rollback](deployment.md) to switch
the service. Context creation is a separate offline operation with higher memory
requirements than cached inference.

See [the evaluation method](fidelity-method.md) for the original input contract,
frozen dataset splits, and held-out validation. External adapted datasets have a
separate [reproduction guide](../reports/external-2026-09-29/README.md).
