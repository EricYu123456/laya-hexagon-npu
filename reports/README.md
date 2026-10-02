# Recorded fidelity experiments

## October 3 precision qualification

[qualification-2026-10-03](qualification-2026-10-03/README.md) preserves the completed
strict-NPU re-evaluation of **all nine historical suites: 3,263 decisions in
1,593 requests**. The [one-percent audit](qualification-2026-10-03/one-percent-audit.json)
passes every suite and required overlapping subset with decision mismatch <=1%,
mean TV <=0.01, identical inputs and zero CPU encoder fallback. The complete typed
split has **12/2,000 differences (0.6000%) and 0.4886% mean TV**; its historical
held-out subset has **11/1,840 (0.5978%) and 0.4884% mean TV**. The largest suite
mean TV is CLINC150's **0.6222%**. Supplementary, synthetic long-input and legacy
suites all have zero decision differences. These are historical regression tests,
not new independent generalization evidence. Aggregate gates do not bound every
answer's error.

A [separate local recomputation](qualification-2026-10-03/independent-audit.json)
matches the Pi audit on all nine suites and four overlapping subset gates, with
the exact frozen plan, input/reference and harness provenance.

The [precision development bundle](precision-development-2026-10-03/README.md)
records the matched 80-decision progression and its cost: mean case latency
increased 84.3%, from 4.764 s to 8.782 s, while the corrected 768 graph grew from
111.36 MiB to 225.25 MiB. The [1024 build bundle](precision-build-2026-10-03/README.md)
retains compiler, cache and swap observations. The raw build-time summaries
retain their pending status from capture; the separate completed qualification
above supersedes that status without changing their recorded evidence bytes.

Fresh [HTTP integration and service acceptance](precision-service-2026-10-03/deployment.json)
passed exact replay of 10 requests / 15 decisions with a **3.593609 GiB** cgroup
peak below 4 GiB, the same PID/invocation, and zero restart/OOM/limit events or CPU
fallback. The selected default manifest remains installed; acceptance restored
the original inactive/dead service state. Earlier service measurements below
belong to the old graph set. See the [current method](../docs/one-percent-qualification.md) for the exact
inventory, overlapping subset gates and offline audit instructions.

## September 29 baseline qualification

[external-2026-09-29](external-2026-09-29/README.md) records the separately frozen 1,160-decision BANKING77/CLINC150/MASSIVE evaluation. **5/5 suites pass**; all per-suite values, paired-language results, input checks, raw outputs, gold-label limitations and probability tails are retained.

[qualification-2026-09-29](qualification-2026-09-29) preserves the completed actual-NPU evaluation. The [declared held-out selection](qualification-2026-09-29/heldout.json) has **65/1,840 decision differences (3.5326%)** and **2.5021% mean TV**. The [complete public split](qualification-2026-09-29/full-public.json) has **71/2,000 differences (3.55%)**. The [predeclared synthetic 1024-token validation](qualification-2026-09-29/long-input-npu.json) has **0/80 differences and 2.0614% mean TV**. All three pass their aggregate thresholds with identical inputs and zero CPU fallbacks.

The [15-decision supplementary development result](qualification-2026-09-29/supplementary-development-npu.json) has **1/15 differences (6.6667%)** and **0.9462% mean TV**; its accuracy gate remains failed. [HTTP integration](qualification-2026-09-29/service-http.json) reproduces those NPU outputs exactly and is reported separately. The service cgroup peak is **2.556 GiB** under its 4 GiB limit, with no restart/OOM or fallback. [Acceptance clarification](qualification-2026-09-29/service-acceptance.md) explains this distinction.

## September 28 development history

[development-2026-09-28](development-2026-09-28) contains development evidence from 16 fixed cases and 80 decisions. The corrected `refined32` candidate reaches the original 5% development thresholds. These are not held-out results or current deployment qualification.

| Report | Provider | Decision mismatch | Mean total variation |
| --- | --- | ---: | ---: |
| [grouped16-npu.json](development-2026-09-28/grouped16-npu.json) | Actual QNN HTP V68 | 12.50% | 8.5510% |
| [grouped16-cpu-diagnostic.json](development-2026-09-28/grouped16-cpu-diagnostic.json) | CPUExecutionProvider | 7.50% | 5.7729% |
| [clsconv16-npu.json](development-2026-09-28/clsconv16-npu.json) | Actual QNN HTP V68 | 20.00% | 18.3522% |
| [clsconv16-cpu-diagnostic.json](development-2026-09-28/clsconv16-cpu-diagnostic.json) | CPUExecutionProvider | 2.50% | 2.8554% |
| [refined32-npu-before-offset-correction.json](development-2026-09-28/refined32-npu-before-offset-correction.json) | Actual QNN HTP V68 | 36.25% | 27.1874% |
| [refined32-corrected-npu.json](development-2026-09-28/refined32-corrected-npu.json) | Actual QNN HTP V68 | **5.00%** | **2.4568%** |

Each report has a sibling `.cases.jsonl` containing original/candidate answers, input identities, timings, and per-question errors. `reference-pi.jsonl` preserves the original Pi CPU reference. `reference-portability.json` compares original Pi CPU and WSL CPU outputs; it does not compare their speed. Build/export sidecars record graph configuration and validation.

Metadata retains the original hosts, package versions, source hashes, and absolute paths. Artifact hashes identify the models actually measured; model binaries and context caches are not included. The corrected development run used a precompiled context with zero CPU fallbacks. See [the evaluation method](../docs/fidelity-method.md) for the fixed data split, metric definitions, reproduction commands, and completed acceptance results. The [held-out selection plan](evaluation-plans/typed-decisions-heldout.json) references the committed exclusion list from before full evaluation. The [deployment summary](qualification-2026-09-29/deployment.json) binds the HTTP report, before/after cgroup snapshots, and 152 passing regression tests to their recorded source revision.
