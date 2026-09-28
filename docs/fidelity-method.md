# Fidelity method and experimental status

Status recorded on 2026-09-29 Asia/Taipei: the corrected actual-HTP candidate passes the declared held-out selection with **3.5326% decision mismatch and 2.5021% mean total variation**. The predeclared synthetic 1024-token validation passes with **0/80 decision differences and 2.0614% mean TV**. HTTP integration and the 4 GiB service memory check pass. A separate 15-decision development suite retains **1/15 differences (6.6667%)** and fails its accuracy gate; this limitation is not hidden by the service result.

## Reference and input contract

The reference is unchanged multilingual Laya from checkpoint revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, downloaded by `download.py`. The checkpoint SHA256 used in these experiments is:

```text
9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204
```

Its configuration specifies `max_len=1024` and `head_max_len=256`. The accelerated agent delegates tokenization, question conversion, sequence construction, marker placement, heads, temperatures, and formatting to the installed original Laya implementation. Only its encoder is replaced. Global attention and local attention remain distinct; the latter uses radius 64, including the boundary. The export uses a finite additive mask penalty of -100, whose effect is checked against original FP32 outputs rather than assumed to be exact for arbitrary logits.

Static buckets add right padding without removing input tokens. The runtime selects a bucket at least as long as the original collated sequence, validates the three float32 ONNX inputs, and fails if no bucket fits. A 768-token graph covers the public split; the deployed set also includes a separately calibrated and validated 1024-token graph for the full original API capacity.

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
| `grouped16` | [Actual HTP V68](../reports/development-2026-09-28/grouped16-npu.json) | 12.50% | 8.5510% | 13.75% |
| `clsconv16` | [CPU QDQ diagnostic](../reports/development-2026-09-28/clsconv16-cpu-diagnostic.json) | 2.50% | 2.8554% | 2.50% |
| `clsconv16` | [Actual HTP V68](../reports/development-2026-09-28/clsconv16-npu.json) | 20.00% | 18.3522% | 28.75% |
| `refined32`, before Conv correction | [CPU QDQ diagnostic](../reports/development-2026-09-28/refined32-cpu-diagnostic.json) | 6.25% | 2.8497% | 5.00% |
| `refined32`, before Conv correction | [Actual HTP V68](../reports/development-2026-09-28/refined32-npu-before-offset-correction.json) | 36.25% | 27.1874% | 42.50% |
| `refined32`, corrected | [Actual HTP V68](../reports/development-2026-09-28/refined32-corrected-npu.json) | **5.00%** | **2.4568%** | **3.75%** |

`grouped16` uses a 768-token graph, U16 activations/U8 weights, grouped GeGLU, 16 calibration sequences, and no CLS split or convolution replacement. Its ONNX SHA256 is:

```text
c0a812181fd2771992a476221d5f66039cdad96dedcb86b23d6102069a016b6b
```

`clsconv16` adds separate CLS/rest residual paths and 1×1 Conv projections with symmetric INT8 weights quantized per output channel, using 32 calibration sequences. Its graph SHA256 is:

```text
c7e56711445411599144f9c6e0f049524c7644be69032727916c95df39cc4da1
```

See the preserved [grouped16 build metadata](../reports/development-2026-09-28/grouped16-build.json), [clsconv16 build metadata](../reports/development-2026-09-28/clsconv16-build.json), and [clsconv16 export configuration](../reports/development-2026-09-28/clsconv16-export.json).

`refined32` adds zero masked padding embeddings and a two-term approximation of each dynamic attention MatMul's 16-bit right operand using supported 8-bit operands. Its uncorrected CPU result still failed to predict HTP behavior. Zero-input probes then found additional constant per-output-channel offsets in 239 of 266 Conv operations; the largest absolute offset was 6.60. Compact spatial-one and three original-shape controls agreed exactly. Adding the negative measured offsets to Conv biases reduced actual whole-model development mismatch from 36.25% to 5.0%. The probes use only zero inputs, not development answers.

The corrected graph SHA256 is `1bbade0d26358f6f3950b2928bb8e4344a170a240fed86e9ef50542c0ab5c138`. Its [correction provenance](../reports/development-2026-09-28/refined32-correction-provenance.json) binds the source graph and measured ORT/QNN backend. The correction intentionally changes CPU graph semantics; running that corrected graph on CPU cannot certify its NPU fidelity. See [the calibration workflow](../npu/CONV_OFFSET_CALIBRATION.md) for shape checks, limitations, and reproduction.

Absolute paths in report metadata identify the original execution environments; the recorded hashes bind the actual graphs and checkpoint. Large model binaries are not bundled with these reports.

An independent [reference portability check](../reports/development-2026-09-28/reference-portability.json) compared pristine ARM Pi CPU and WSL x86 CPU outputs on these 80 development decisions: zero decision mismatches and mean TV `9.4178e-7`. This supports using the captured FP32 outputs for numerical diagnostics on this selection, but says nothing about cross-host performance equivalence. The complete 400-case FP32 reference was captured in WSL; the preserved [Pi reference](../reports/development-2026-09-28/reference-pi.jsonl) covers the development cases.

## Independent hardware qualification

The frozen corrected-768 graph ran all 400 cases on the Pi. The [full report](../reports/qualification-2026-09-29/full-public.json) records 2,005 NPU calls including five warmup questions, one cached QNN session, zero CPU fallbacks, and identical input sequences/markers for all cases. The [held-out report](../reports/qualification-2026-09-29/heldout.json) is derived offline from those exact outputs using the exclusion list committed before the run; no additional inference or selection by observed errors is involved.

| Selection | Decisions | Decision differences | Decision mismatch | Mean TV | Argmax mismatch |
| --- | ---: | ---: | ---: | ---: | ---: |
| Declared held-out | 1,840 | 65 | **3.5326%** | **2.5021%** | 3.9130% |
| Complete public split, including calibration/development | 2,000 | 71 | **3.5500%** | **2.5027%** | 3.8500% |

The held-out mean probability MAE is 1.4731 percentage points. TV's 95th percentile is 6.8687%, and its maximum is 32.87%; the aggregate pass is not a per-prediction error guarantee. The full run's median latency is 4,764 ms per five-question case on the Pi. Its stored CPU reference ran on WSL and must not be used to claim hardware speedup.

The qualification folder preserves the unchanged CPU reference and full/held-out per-case records. To recompute the held-out report without NPU inference:

```bash
python benchmark_fidelity.py --dataset-path "$DATA" \
  --from-candidate-records reports/qualification-2026-09-29/full-public.cases.jsonl \
  --source-report reports/qualification-2026-09-29/full-public.json \
  --reference reports/qualification-2026-09-29/reference-wsl.jsonl \
  --selection-plan reports/evaluation-plans/typed-decisions-heldout.json \
  --output .work/recomputed-heldout.json
```

The helper checks the committed declaration's Git identity, dataset and input hashes, exact original reference, and stored per-question metrics. Derived reports retain the original run's counters under `original_run`; they do not invent subset NPU counters.

## Reproduction and evaluation commands

Follow the Python 3.12 setup, pinned checkpoint/dataset download, and corrected candidate build commands in [README.md](../README.md). Use [requirements-fidelity-build.txt](../requirements-fidelity-build.txt) for x86/WSL builds, not the target environment's requirements. Copy the graph directory and manifest together to the Pi. Keep recipes in separate directories because a manifest's bucket mapping alone is not a complete recipe description. Preserve the export sidecar and quantization metadata.

For the earlier `grouped16` experiment, use only `--group-outliers --samples 16`. `clsconv16` adds `--conv-linear --split-cls --samples 32` but omits zero-padding, MatMul refinement, and measured offset correction. The qualified 768-token recipe does not fold norms. The separately selected 1024-token recipe adds `--fold-norms` and `--append-long-calibration`; `--balance` is not part of either recipe.

`npu/split_residual.py` is an unintegrated experimental alternative that separates residual feature bands and rescales before LayerNorm. Its fixed scaled epsilon is an approximation. It has CPU checks only and is not part of the measured or deployed recipe.

For every target QNN run:

```bash
source .venv/bin/activate
source npu/env.sh
export LAYA_NPU_MANIFEST="$PWD/.work/corrected_768/manifest.json"
export LAYA_NPU_CONTEXT_CACHE=1
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

After freezing a new hardware candidate, run the full declared held-out selection. The result above used a full-split run followed by the verified offline derivation; this command evaluates only the declared held-out cases directly:

```bash
EXCLUDED=0,1,25,26,50,51,75,76,100,101,125,126,150,151,175,176,200,201,225,226,250,251,275,276,300,301,325,326,350,351,375,376
python benchmark_fidelity.py --backend npu --dataset-path "$DATA" \
  --reference .work/reference-all.jsonl --exclude-indices "$EXCLUDED" \
  --threads 4 --output .work/candidate-heldout.json
```

Use a fresh output filename for each run. The script refuses to overwrite evidence. A threshold failure returns exit code 1; inspect both the aggregate JSON and sibling `.cases.jsonl`. `selected_cases_pass` alone does not mean a full or held-out pass. Confirm `heldout_selection_pass`, complete selection, input equality, the actual provider, and zero CPU fallbacks.

## Long-input validation design

The public split's longest original sequence has 631 tokens, so its pass does not establish behavior at 1024 tokens. The 1024-token candidate was selected using only the fixed 15-decision Chinese/English supplementary probes. Its CPU QDQ diagnostic has zero decision differences and 0.8245% mean TV; these numbers do not qualify the NPU graph. The selected source hash and allowed zero-input HTP correction are recorded in the [candidate declaration](../reports/evaluation-plans/long-input-candidate-static-gamma.json).

The initial folded graph failed strict HTP compilation because the exporter/quantizer represented constant-one LayerNorm gamma as U16 aliases. A [small hardware probe](../reports/development-2026-09-28/unit-layernorm-htp.json) verifies that static U8 gamma and a matching I32 zero-bias encoding are accepted, with bit-identical CPU outputs. `npu/unit_layernorm.py` materializes only those exact constant parameters; it rejects dynamic/non-unit gamma and nonzero beta. It does not alter calibration ranges, activations, or Conv weights. Whole-model NPU validation is still required after this compatibility repair.

A separate [predeclared validation plan](../reports/evaluation-plans/long-input-heldout.json) selects source rows `2,27,...,377`, excluding every calibration and development row. `make_long_probe_suite.py` repeats each selected state while keeping its questions, criteria, tokenizer, sequence construction, and original budgets unchanged, until all 80 questions occupy 1024 tokens. These are synthetic stress inputs; their untransformed source rows were already present in the completed 768-token evaluation. They are not an independent natural-language dataset. Their transformed reference answers are excluded from 1024 model selection.

To reproduce the additional validation with fresh output paths:

```bash
python make_long_probe_suite.py --dataset-path "$DATA" \
  --output .work/long-validation-suite.json
python benchmark_probes.py --suite .work/long-validation-suite.json \
  --backend cpu --threads 4 --output .work/long-reference.json
python benchmark_probes.py --suite .work/long-validation-suite.json \
  --reference .work/long-reference.json --threads 4 --output .work/long-npu.json
```

For the final command, load the target QNN environment and point `LAYA_NPU_MANIFEST` at the corrected graph set. In addition to the 5% decision/mean-TV gates, the harness checks exact token/marker identities and verifies the expected NPU bucket-call increment separately for every request.

The final corrected 1024 graph SHA256 is `cfd85315c4ee6b5bfaeb24ced44bba3cf6ebb37ff421efb4c65ade90c571ec64`. Its [frozen declaration](../reports/evaluation-plans/long-input-candidate-final.json) preceded NPU validation. A fresh build reproduces both its source and corrected graph byte-for-byte using the saved zero-input measurements.

The [80-decision NPU result](../reports/qualification-2026-09-29/long-input-npu.json) has zero decision/argmax differences, mean TV **2.0614%**, identical inputs, and exactly 80 calls to the 1024 bucket with zero fallback/cache errors. TV's 95th percentile is 5.7022% and maximum 11.5332%; this is an aggregate pass, not an individual-probability guarantee.

The separate [15-decision development report](../reports/qualification-2026-09-29/supplementary-development-npu.json) retains one long-English choice difference, or **6.6667% mismatch**, with mean TV **0.9462%** and maximum TV **4.23%**. Its `passed` flag remains false. The graph was not changed after this result. The original held-out and synthetic long-input criteria remain unchanged. [The service acceptance clarification](../reports/qualification-2026-09-29/service-acceptance.md) records why exact HTTP/CLI replay is an integration check rather than another fidelity pass.

## Runtime, cache, and performance boundaries

Strict NPU means the encoder runs through QNN HTP with `session.disable_cpu_ep_fallback=1`; CPU tokenization, embedding lookup, and original decision heads remain intentional parts of the API. `session.disable_fallback()` also prevents Python's provider-error retry. A CPU QDQ adapter supplied through `--module` is a distinct diagnostic and must identify its provider in the report.

`npu/env.sh` selects the installed QNN wheel's backend/stub/skeleton and the system DSP support libraries, and clears the conflicting `DSP_LIBRARY_PATH`. This environment is required for both standalone commands and the service wrapper. It is unrelated to the x86 build environment. The benchmark records package and source versions for reproducibility.

The manifest resolves graph paths relative to its directory and records checkpoint/graph hashes and mask policy. Loading verifies those identities. Context cache keys additionally bind runtime versions and provider options; cache entries are embedded single-file ONNX contexts published atomically. Invalid entries fail with an actionable error instead of falling back to CPU. QNN's context options are documented in [ORT's EPContext design](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html).

`npu/compile_contexts.py --manifest <path> --verify-reload` prepares buckets without loading checkpoint tensors or token embeddings, then releases the preparation adapter before reloading saved contexts. Set `LAYA_NPU_CONTEXT_CACHE=1` and optionally set `LAYA_NPU_CACHE_DIR`. The initial corrected-768 cache compiled in 419.5 seconds and reloaded in 1.62 seconds. After upgrading the cache key to schema 2, preparation took 442.4 seconds and reload took 1.43 seconds; both saved graphs contain a single QNN EPContext. A [five-question equivalence replay](../reports/qualification-2026-09-29/cache-v2-equivalence.json) confirms unchanged outputs after the schema upgrade. Cache schema changes require preparation again. A cache hit alone is not a fidelity test.

Latency reports include tokenization, heads, and inference but exclude model loading and warmup. Compare speed only with the same host, CPU thread count, original input contract, and measured selection. Compilation, cache reload, warm latency, bucket switching, and peak memory need separate measurements. The 1024 compile exhausted physical RAM on its first attempt; the identical retry used temporary 4 GiB swap and reached 5.73 GiB peak RSS. That swap was removed before a fresh-process cache load, which took 2.19 seconds and peaked at 1.18 GiB RSS. [Cache evidence](../reports/qualification-2026-09-29/cache-v2-1024.json) preserves both the compilation and no-swap reload checks.

The deployed [HTTP verification](../reports/qualification-2026-09-29/service-http.json) passed exact replay of all 15 standalone NPU answers, five invalid-input checks, and both bucket transitions. Startup plus this workload peaked at **2,744,172,544 bytes (2.556 GiB)** in the actual 4 GiB cgroup, with the same PID/invocation and zero restarts, limit events, or OOMs. Health reported 16 NPU calls including startup warmup, five context hits, and zero fallback/cache misses/errors. The separate development-fidelity flag remains false. See [deployment instructions](deployment.md) for reproduction. Historical results in `npu/REPORT.md` describe the earlier truncated/clamped implementation and do not qualify the current graph set.
