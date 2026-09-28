# Fidelity method and experimental status

Status recorded on 2026-09-28: **the ≤5% target is not achieved**. The best measured actual-HTP candidate has 12.5% decision mismatch and 8.551% mean total variation on the development selection. No held-out pass, validated 1024-token model set, or qualified service deployment is claimed.

## Reference and input contract

The reference is unchanged multilingual Laya from checkpoint revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, downloaded by `download.py`. The checkpoint SHA256 used in these experiments is:

```text
9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204
```

Its configuration specifies `max_len=1024` and `head_max_len=256`. The accelerated agent delegates tokenization, question conversion, sequence construction, marker placement, heads, temperatures, and formatting to the installed original Laya implementation. Only its encoder is replaced. Global attention and local attention remain distinct; the latter uses radius 64, including the boundary. The export uses a finite additive mask penalty of -100, whose effect is checked against original FP32 outputs rather than assumed to be exact for arbitrary logits.

Static buckets add right padding without removing input tokens. The runtime selects a bucket at least as long as the original collated sequence, validates the three float32 ONNX inputs, and fails if no bucket fits. A 768-token graph is sufficient for the measured development selection but does not implement the full 1024-token API capacity. That final capacity and its hardware behavior still require validation.

The old runtime imposed 64-token sequence and head budgets and clamped GeGLU values to ±50. These changes invalidate a claim of equivalence with the original model. The new grouped representation preserves every channel and weight: it separates gate, value, and output projections for outlier channels and sums their contributions. Mathematical equivalence is checked before quantization. Those FP32 checks do not imply that HTP's quantized implementation will agree with CPU QDQ.

## Dataset and fixed selections

The benchmark uses [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), file `all/test-00000-of-00001.parquet`, pinned to revision:

```text
c76749ec58bd8c3d2ea706b31c333a9059c38f90
```

The required parquet SHA256 is:

```text
4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c
```

`benchmark_fidelity.py` verifies this hash even for a local dataset. The split contains 400 cases and 2,000 decisions. Zero-based row selections were fixed before candidate evaluation:

| Purpose | Selection | Cases | Decisions |
| --- | --- | ---: | ---: |
| Calibration | `i % 25 == 0` | 16 | 80 available |
| Development/model selection | `i % 25 == 1` | 16 | 80 |
| Declared held-out evaluation | All other rows | 368 | 1,840 |

The builder's `--samples` counts calibration **question sequences**, not cases. `grouped16` used 16 complete sequences spread across the calibration cases; `clsconv16` used 32. A bucket excludes calibration sequences that do not fit rather than truncating them. Per-build metadata records the actual selected cases and sample count.

Development data may guide changes. Held-out outputs must not guide calibration, graph changes, or model selection. Freeze the recipe before evaluating all 368 held-out cases. If results are used for further tuning, disclose that reuse and obtain a fresh independent validation set before making a held-out claim. A full-public-split report includes calibration and development cases and is labeled separately.

## Pass criteria and reported errors

The intended gate requires all three conditions on the complete declared held-out selection:

1. Decision mismatch at most 5%.
2. Mean total variation at most 0.05.
3. Identical original and candidate token sequences and marker positions for every question.

Decision mismatch follows the API's output semantics: exact choice label, `noul > 0.5`, or Python `round()` of the expected score. Argmax mismatch is also reported, separately from expected-score decisions. Total variation is `0.5 * sum(abs(reference_probability - candidate_probability))` per question, averaged over decisions. For `noul`, the distribution is `[1-p, p]`. API-rounded distributions are normalized before comparison. Here “5% TV” means 0.05 of probability mass, not relative error divided by a possibly tiny reference probability.

Reports include per-type and per-workflow results, tail probability errors, normalized score error, confidence/action-probability errors, artifact hashes, source hashes, and accelerator counters. This evaluates fidelity to Laya, not correctness against gold labels. Passing a finite test selection is not a guarantee for every possible input.

## Measured development results

Every row below uses the same 16 development cases/80 decisions and identical preprocessed inputs. CPU QDQ runs use ONNX Runtime's `CPUExecutionProvider`; they are diagnostic results and cannot qualify an NPU candidate. Links point to the preserved reports, with sibling `.cases.jsonl` files containing per-case evidence.

| Candidate | Execution/report | Decision mismatch | Mean TV | Argmax mismatch |
| --- | --- | ---: | ---: | ---: |
| `grouped16` | [CPU QDQ diagnostic](../reports/development-2026-09-28/grouped16-cpu-diagnostic.json) | 7.50% | 5.7729% | 11.25% |
| `grouped16` | [Actual HTP V68](../reports/development-2026-09-28/grouped16-npu.json) | **12.50%** | **8.5510%** | 13.75% |
| `clsconv16` | [CPU QDQ diagnostic](../reports/development-2026-09-28/clsconv16-cpu-diagnostic.json) | 2.50% | 2.8554% | 2.50% |
| `clsconv16` | [Actual HTP V68](../reports/development-2026-09-28/clsconv16-npu.json) | 20.00% | 18.3522% | 28.75% |

`grouped16` uses a 768-token graph, U16 activations/U8 weights, grouped GeGLU, 16 calibration sequences, and no CLS split or convolution replacement. Its ONNX SHA256 is:

```text
c0a812181fd2771992a476221d5f66039cdad96dedcb86b23d6102069a016b6b
```

`clsconv16` adds separate CLS/rest residual paths and 1×1 Conv projections with symmetric INT8 weights quantized per output channel, using 32 calibration sequences. Its graph SHA256 is:

```text
c7e56711445411599144f9c6e0f049524c7644be69032727916c95df39cc4da1
```

See the preserved [grouped16 build metadata](../reports/development-2026-09-28/grouped16-build.json), [clsconv16 build metadata](../reports/development-2026-09-28/clsconv16-build.json), and [clsconv16 export configuration](../reports/development-2026-09-28/clsconv16-export.json).

The CPU-to-HTP difference is unresolved. The lower CPU QDQ error did not predict a better hardware model. Neither candidate passes the hardware target. Absolute paths in report metadata identify the original execution environments; the recorded hashes bind the actual graphs and checkpoint. Large model binaries are not bundled with these reports.

An independent [reference portability check](../reports/development-2026-09-28/reference-portability.json) compared pristine ARM Pi CPU and WSL x86 CPU outputs on these 80 development decisions: zero decision mismatches and mean TV `9.4178e-7`. This supports using the captured FP32 outputs for numerical diagnostics on this selection, but says nothing about cross-host performance equivalence. The complete 400-case FP32 reference was captured in WSL; the preserved [Pi reference](../reports/development-2026-09-28/reference-pi.jsonl) covers the development cases.

## Reproduction and evaluation commands

Follow the Python 3.12 setup, pinned checkpoint/dataset download, and `grouped16` build commands in [README.md](../README.md). Use [requirements-fidelity-build.txt](../requirements-fidelity-build.txt) for x86/WSL builds, not the target environment's requirements. Copy the graph directory and manifest together to the Pi. Keep recipes in separate directories because a manifest's bucket mapping alone is not a complete recipe description. Preserve the export sidecar and quantization metadata.

To reproduce the failed `clsconv16` experiment, use the same build command with `--samples 32 --conv-linear --split-cls` and a separate `--output-dir .work/clsconv16`. Do not deploy it based on its CPU diagnostic pass. `--fold-norms` and `--balance` are additional experimental transformations, not part of either tabled recipe.

For every target QNN run:

```bash
source .venv/bin/activate
source npu/env.sh
export LAYA_NPU_MANIFEST="$PWD/.work/grouped16/manifest.json"
export LAYA_NPU_CONTEXT_CACHE=0
DATA=.work/dataset/all/test-00000-of-00001.parquet
DEV=1,26,51,76,101,126,151,176,201,226,251,276,301,326,351,376
```

Capture a complete reference with unchanged Laya. Running this on the Pi also supplies a same-host latency baseline; WSL output can support numerical comparisons but not a Pi speedup claim:

```bash
python benchmark_fidelity.py --backend cpu --dataset-path "$DATA" \
  --threads 4 --output .work/reference-all.jsonl
# If interrupted, resume the same reference without changing its environment:
python benchmark_fidelity.py --backend cpu --dataset-path "$DATA" \
  --threads 4 --output .work/reference-all.jsonl --resume
```

During model selection, compare only development cases:

```bash
python benchmark_fidelity.py --backend npu --dataset-path "$DATA" \
  --reference .work/reference-all.jsonl --indices "$DEV" --threads 4 \
  --output .work/candidate-development.json
```

After freezing a hardware candidate, run the full declared held-out selection. **This is a future qualification step; no passing result is recorded yet.** It may require additional validated buckets:

```bash
EXCLUDED=0,1,25,26,50,51,75,76,100,101,125,126,150,151,175,176,200,201,225,226,250,251,275,276,300,301,325,326,350,351,375,376
python benchmark_fidelity.py --backend npu --dataset-path "$DATA" \
  --reference .work/reference-all.jsonl --exclude-indices "$EXCLUDED" \
  --threads 4 --output .work/candidate-heldout.json
```

Use a fresh output filename for each run. The script refuses to overwrite evidence. A threshold failure returns exit code 1; inspect both the aggregate JSON and sibling `.cases.jsonl`. `selected_cases_pass` alone does not mean a full or held-out pass. Confirm `heldout_selection_pass`, complete selection, input equality, the actual provider, and zero CPU fallbacks.

## Runtime, cache, and performance boundaries

Strict NPU means the encoder runs through QNN HTP with `session.disable_cpu_ep_fallback=1`; CPU tokenization, embedding lookup, and original decision heads remain intentional parts of the API. `session.disable_fallback()` also prevents Python's provider-error retry. A CPU QDQ adapter supplied through `--module` is a distinct diagnostic and must identify its provider in the report.

`npu/env.sh` selects the installed QNN wheel's backend/stub/skeleton and the system DSP support libraries, and clears the conflicting `DSP_LIBRARY_PATH`. This environment is required for both standalone commands and the service wrapper. It is unrelated to the x86 build environment. The benchmark records package and source versions for reproducibility.

The manifest resolves graph paths relative to its directory and records checkpoint/graph hashes and mask policy. Loading verifies those identities. Context cache keys additionally bind runtime versions and provider options; cache entries are embedded single-file ONNX contexts published atomically. Invalid entries fail with an actionable error instead of falling back to CPU. QNN's context options are documented in [ORT's EPContext design](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html).

`npu/compile_contexts.py --manifest <path> --verify-reload` prepares buckets without loading checkpoint tensors or token embeddings, then releases the preparation adapter before reloading saved contexts. Set `LAYA_NPU_CONTEXT_CACHE=1` to test it and optionally set `LAYA_NPU_CACHE_DIR`. Cache configuration and failure handling pass mock tests; target generation/reload validation remains pending. A cache hit is not a fidelity test.

Latency reports include tokenization, heads, and inference but exclude model loading and warmup. Compare speed only with the same host, CPU thread count, original input contract, and measured selection. Compilation, cache reload, warm latency, bucket switching, and peak memory need separate measurements. Precompilation matters because loading full weights alongside graph compilation can exceed the service's 4 GiB memory limit; hardware memory qualification is still required.

The remaining acceptance work is to resolve the CPU-QDQ/HTP numerical gap, pass independent held-out evaluation, validate buckets through the original 1024-token budget, verify context reload on hardware, and exercise the service under its actual memory limit. Historical results in `npu/REPORT.md` describe the earlier truncated/clamped implementation and do not satisfy these criteria.
