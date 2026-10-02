# Fidelity method and experimental status

Status recorded on 2026-10-03 Asia/Taipei: the precision graph set passes **all nine historical NPU regression suites and every required subset**, each with decision mismatch <=1% and mean total variation <=0.01. The [completed audit](../reports/qualification-2026-10-03/one-percent-audit.json) verifies unchanged inputs, the original checkpoint and strict HTP encoder execution with zero CPU fallback. Fresh [HTTP integration and service cgroup acceptance](../reports/precision-service-2026-10-03/deployment.json) also pass, with a 3.593609 GiB peak below 4 GiB. The selected default manifest remains installed; the service was restored to its original inactive/dead state after testing.

## Current historical-regression qualification

The frozen [qualification plan](../reports/qualification-2026-10-03/qualification-plan.json) covers 1,593 requests and 3,263 decisions. A [separate local audit](../reports/qualification-2026-10-03/independent-audit.json) reproduces the Pi audit's pass for all nine suites and four overlapping subsets, including the same frozen plan and 3,263 NPU calls. The [one-percent method](one-percent-qualification.md) fixes the complete historical inventory and recomputes errors from original and candidate answers. Older low-level reports' 5% pass flags do not determine acceptance.

| Required suite | Decision differences | Mean TV |
| --- | ---: | ---: |
| [Typed-decisions complete public split](../reports/qualification-2026-10-03/typed-decisions-full-npu.json) | 12/2,000 (0.6000%) | 0.4886% |
| [Supplementary multilingual/long/switching](../reports/qualification-2026-10-03/supplementary-development-npu.json) | 0/15 | 0.2205% |
| [Synthetic 1024-token validation](../reports/qualification-2026-10-03/long-input-npu.json) | 0/80 | 0.4848% |
| [BANKING77, eight intents](../reports/qualification-2026-10-03/banking77-card8-npu.json) | 1/320 (0.3125%) | 0.4265% |
| [CLINC150, ten domains](../reports/qualification-2026-10-03/clinc150-domain10-npu.json) | 1/300 (0.3333%) | 0.6222% |
| [MASSIVE en-US](../reports/qualification-2026-10-03/massive-en-US-npu.json) | 0/180 | 0.4108% |
| [MASSIVE zh-TW](../reports/qualification-2026-10-03/massive-zh-TW-npu.json) | 1/180 (0.5556%) | 0.5205% |
| [MASSIVE zh-CN](../reports/qualification-2026-10-03/massive-zh-CN-npu.json) | 0/180 | 0.4675% |
| [Legacy accuracy/service requests](../reports/qualification-2026-10-03/legacy-npu.json) | 0/8 | 0.2087% |

The overlapping historical typed held-out subset separately passes with **11/1,840 differences (0.5978%) and 0.4884% mean TV**. Typed development has 0/80 differences and 0.5770% mean TV; each four-decision legacy subset has zero differences. Subsets are not counted again in the executed total. The five external suites have a descriptive pooled 3/1,160 differences and 0.4956% mean TV, but each individual gate remains required.

This is a re-evaluation of historical inputs, **not a fresh independent generalization claim**. MASSIVE languages share 180 semantic IDs, and synthetic long inputs derive from typed source cases. The recipe was selected on reserved development inputs before these final runs. Aggregate passes do not bound every answer: full typed TV has P95 1.47% and maximum 10.16%.

The [frozen manifest](../reports/qualification-2026-10-03/candidate-manifest.json) binds corrected graphs `a3ba4ac988f726d61d06543e723612e63c3047fe53d9133b8a7b5858a7365003` (768) and `85ce822364a7645a450c0925e5e6ead066edeac8e298d14e22caf31575938843` (1024), including their HTP correction provenance. Large graph/context binaries are not included in Git.

## Reference and input contract

The reference is unchanged multilingual Laya from checkpoint revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, downloaded by `download.py`. The checkpoint SHA256 used in these experiments is:

```text
9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204
```

Its configuration specifies `max_len=1024` and `head_max_len=256`. The accelerated agent delegates tokenization, question conversion, sequence construction, marker placement, heads, temperatures, and formatting to the installed original Laya implementation. Only its encoder is replaced. Global attention and local attention remain distinct; the latter uses radius 64, including the boundary. The export uses a finite additive mask penalty of -100, whose effect is checked against original FP32 outputs rather than assumed to be exact for arbitrary logits.

Static buckets add right padding without removing input tokens. The runtime selects a bucket at least as long as the original collated sequence, validates the three float32 ONNX inputs, and fails if no bucket fits. A 768-token graph covers the public split; the qualified set also includes a separately calibrated and validated 1024-token graph for the full original API capacity.

The old runtime imposed 64-token sequence and head budgets and clamped GeGLU values to ±50. These changes invalidate a claim of equivalence with the original model. The current representation retains all channels and original projection contributions, separates GeGLU outliers and CLS/rest residual feature bands, folds compatible norms, and refines weights using two INT8 terms. Three remaining learned LayerNorm affines use U16 elementwise arithmetic. Per-token scaling before LayerNorm uses a fixed epsilon approximation, so the builder requires maximum FP32 export error below 0.01 before quantization. Those checks do not imply HTP agreement with CPU QDQ; whole-model NPU qualification remains required.

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
| Historical held-out evaluation | All other rows | 368 | 1,840 |

The builder's `--samples` counts calibration **question sequences**, not cases. The current 768 recipe uses the same 32 reserved sequences as `refined32`; 1024 retains those originals and appends 32 full-length variants from the same cases. The earlier `grouped16` used 16 sequences and `clsconv16` used 32. A bucket excludes sequences that do not fit rather than truncating them. Per-build metadata records exact identities and sample counts.

Development data may guide changes. Final evaluation outputs must not guide calibration, graph changes, or model selection. The historical held-out name records the original split; the present re-run is a regression test. If evaluation results guide further tuning, disclose that reuse and obtain a fresh independent dataset before making an independent validation claim. A full-public-split report includes calibration and development cases and is labeled separately.

## Pass criteria and reported errors

The current gate requires all conditions on every complete historical suite and required subset:

1. Decision mismatch at most 1%.
2. Mean total variation at most 0.01.
3. Identical original and candidate token sequences and marker positions for every question.
4. Verified original model/input identity and strict HTP execution with zero CPU fallback.

Decision mismatch follows the API's output semantics: exact choice label, `noul > 0.5`, or Python `round()` of the expected score. Argmax mismatch is also reported, separately from expected-score decisions. Total variation is `0.5 * sum(abs(reference_probability - candidate_probability))` per question, averaged over decisions. For `noul`, the distribution is `[1-p, p]`. API-rounded distributions are normalized before comparison. Here “1% TV” means 0.01 of probability mass, not relative error divided by a possibly tiny reference probability. Historical September reports retain their original 5% criteria.

Reports include per-type and per-workflow results, tail probability errors, normalized score error, confidence/action-probability errors, artifact hashes, source hashes, and accelerator counters. This evaluates fidelity to Laya, not correctness against gold labels. Passing a finite test selection is not a guarantee for every possible input.

## Historical September 28 development results

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

## Historical September 29 hardware qualification

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

Follow the Python 3.12 setup, pinned checkpoint/dataset download, and corrected candidate build commands in [the build guide](build-fidelity.md). Use [requirements-fidelity-build.txt](../requirements-fidelity-build.txt) for x86/WSL builds, not the target environment's requirements. Copy the graph directory and manifest together to the Pi. Keep recipes in separate directories because a manifest's bucket mapping alone is not a complete recipe description. Preserve the export sidecar and quantization metadata.

The selected precision recipe uses `--split-cls --split-features --feature-thresholds 100,500,2000 --fold-norms` on both buckets, then the permanent weight and affine refinement CLIs before graph-specific HTP Conv-offset calibration. Its feature groups contain 631, 127, 6 and 4 channels. The 1024 build additionally uses `--append-long-calibration`; `--balance` and experimental alternative feature partitions are not selected. Follow the complete command sequence in the build guide.

`npu/split_residual.py` is integrated into this recipe. It retains feature order and full attention while separating residual quantization ranges, including distinct CLS/rest paths within each feature band. Its scaled-epsilon approximation passes the builder's FP32 gate, and the complete refined graph is qualified on actual HTP. The [precision development bundle](../reports/precision-development-2026-10-03/README.md) records the selected 768 result and its performance tradeoff; it was captured while full qualification was still pending.

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

After freezing a new candidate, use the [complete one-percent prepare/run/audit workflow](one-percent-qualification.md). A typed-only run cannot establish all-suite acceptance. The following command reproduces just the historical typed held-out subset:

```bash
EXCLUDED=0,1,25,26,50,51,75,76,100,101,125,126,150,151,175,176,200,201,225,226,250,251,275,276,300,301,325,326,350,351,375,376
python benchmark_fidelity.py --backend npu --dataset-path "$DATA" \
  --reference .work/reference-all.jsonl --exclude-indices "$EXCLUDED" \
  --threads 4 --output .work/candidate-heldout.json
```

Use a fresh output filename for each run. The script refuses to overwrite evidence. Low-level benchmark pass flags retain their own thresholds; only the complete one-percent audit establishes current acceptance. Confirm exact coverage, original reference and graph identities, input equality, the actual provider, and zero CPU fallbacks.

## Long-input design and historical September 29 result

The current precision 1024 graph retains the original 1024/256-token budgets and passes the same synthetic suite with 0/80 decision differences and 0.4848% mean TV. The current supplementary suite passes 0/15 with 0.2205% mean TV. The following selection history and old results describe the earlier graph, not the current candidate.

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

## External adaptation and historical September 29 result

The current regression re-evaluation of these same five suites passes each 1% gate; the individual values and artifact hashes are in the current qualification section above. The following evidence describes their original September evaluation and preserves its original thresholds.

A later [frozen external evaluation](../reports/evaluation-plans/external-datasets-2026-09-29.json)
uses BANKING77 (eight fixed intents, 320 decisions), CLINC150 (ten domains, 300),
and MASSIVE English/Traditional Chinese/Simplified Chinese (180 paired IDs per
language). The unchanged deployed graph set passes all five suites separately:
decision mismatch ranges from **1.6667% to 3.8889%**, and mean TV from **2.2582%
to 2.8311%**. The pooled descriptive result is **34/1,160 (2.9310%)** mismatch and
**2.5973% mean TV**, with exactly 1,160 NPU calls and zero CPU fallback.

These are fixed subsets and adapted choice-label spaces, not the datasets'
native leaderboard tasks. Source revisions, sampling rules, original reference
outputs, full input-integrity checks, each suite's individual gate, paired-language
comparisons, gold-label limitations, and error tails are documented in the
[external report](../reports/external-2026-09-29/README.md). No calibration or model
selection used these results. Inputs are 158–233 tokens; this supplements the
preceding long-input and mixed-head tests. The exact plan was committed before
either CPU or NPU inference.

## Runtime, cache, and performance boundaries

Strict NPU means the encoder runs through QNN HTP with `session.disable_cpu_ep_fallback=1`; CPU tokenization, embedding lookup, and original decision heads remain intentional parts of the API. `session.disable_fallback()` also prevents Python's provider-error retry. A CPU QDQ adapter supplied through `--module` is a distinct diagnostic and must identify its provider in the report.

`npu/env.sh` selects the installed QNN wheel's backend/stub/skeleton and the system DSP support libraries, and clears the conflicting `DSP_LIBRARY_PATH`. This environment is required for both standalone commands and the service wrapper. It is unrelated to the x86 build environment. The benchmark records package and source versions for reproducibility.

The manifest resolves graph paths relative to its directory and records checkpoint/graph hashes and mask policy. Loading verifies those identities. Context cache keys additionally bind runtime versions and provider options; cache entries are embedded single-file ONNX contexts published atomically. Invalid entries fail with an actionable error instead of falling back to CPU. QNN's context options are documented in [ORT's EPContext design](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html).

`npu/compile_contexts.py --manifest <path> --verify-reload` prepares buckets without loading checkpoint tensors or token embeddings, then releases the preparation adapter before reloading saved contexts. Set `LAYA_NPU_CONTEXT_CACHE=1` and optionally set `LAYA_NPU_CACHE_DIR`. For the September 29 baseline, the initial corrected-768 cache compiled in 419.5 seconds and reloaded in 1.62 seconds. After upgrading the cache key to schema 2, preparation took 442.4 seconds and reload took 1.43 seconds; both saved graphs contain a single QNN EPContext. A [five-question equivalence replay](../reports/qualification-2026-09-29/cache-v2-equivalence.json) confirms unchanged baseline outputs after the schema upgrade. Cache schema changes require preparation again. A cache hit alone is not a fidelity test.

Latency reports must state whether model loading and warmup are excluded. Compare speed only with the same host, CPU thread count, original input contract, and measured selection. Compilation, cache reload, warm latency, bucket switching, and peak memory need separate measurements. The September 29 baseline's 1024 compile exhausted physical RAM on its first attempt; the identical retry used temporary 4 GiB swap and reached 5.73 GiB peak RSS. That swap was removed before a fresh-process cache load, which took 2.19 seconds and peaked at 1.18 GiB RSS. Historical [cache evidence](../reports/qualification-2026-09-29/cache-v2-1024.json) preserves those checks.

The September 29 [HTTP verification](../reports/qualification-2026-09-29/service-http.json) passed exact replay of all 15 standalone NPU answers, five invalid-input checks, and both bucket transitions. Its **2,744,172,544-byte (2.556 GiB)** service cgroup peak, unchanged PID/invocation, zero restarts/OOMs and failed supplementary-fidelity flag all describe that baseline graph. Historical results in `npu/REPORT.md` describe the still earlier truncated/clamped implementation.

For the current precision recipe, matched 80-decision Pi development measurements increased mean case latency from 4.764 s to 8.782 s (**84.3%**) and corrected 768 graph size from 111.36 MiB to 225.25 MiB. The [768 compile/reload evidence](../reports/precision-development-2026-10-03/README.md) records 6.54 GiB peak process RSS and 9.75 s verified reload; the [1024 build evidence](../reports/precision-build-2026-10-03/README.md) records 6.59 GiB and 12.59 s. Temporary build swap was removed before qualification. These measurements are not a new service cgroup peak or a cross-host CPU speed comparison.

The separate [precision service acceptance](../reports/precision-service-2026-10-03/deployment.json) passed exact HTTP replay of 10 requests / 15 decisions, invalid-input checks and both bucket transitions. Peak cgroup memory was **3,858,608,128 bytes (3.593609 GiB)** under 4 GiB, with the same PID/invocation and zero restarts, OOMs or memory-limit events. NPU execution had zero CPU fallbacks or cache misses/errors/writes. Recorded swap snapshots were zero and both temporary build swap files were absent; no peak swap usage is claimed. After verification the service returned to inactive/dead, while the selected default manifest stayed installed. See [deployment instructions](deployment.md) and the [HTTP evidence](../reports/precision-service-2026-10-03/service-http.json).
