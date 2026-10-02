# One-percent historical qualification — 2026-10-03

**PASS:** all nine historical suites and all four predeclared overlapping subsets
independently meet **decision mismatch ≤1% and mean total variation (TV) ≤0.01**.
The run covers **1,593 requests and 3,263 decisions**, with **3,263 QNN HTP encoder
calls and zero CPU EP fallbacks**. Every suite has complete input/output and
per-request execution evidence; no suite was omitted or rescued by pooling.

The [Pi audit](one-percent-audit.json) and a fresh
[independent local offline audit](independent-audit.json) agree on all metrics and
report hashes. The latter was created at `2026-10-02T19:38:07.271192+00:00`. The immutable
[plan](qualification-plan.json) froze at `2026-10-02T17:51:09.932811+00:00`
(2026-10-03 in Asia/Taipei), before this candidate's validation inference.

## Results

All rows below pass. TV is shown as a percentage (`100 × TV`); the mean-TV limit
is therefore **1.000000%**. Values are rounded here; JSON reports retain full
precision. The p95 and maximum columns describe individual-decision errors and
are not additional pass thresholds.

| Suite | Requests | Decisions | Mismatches | Mismatch % | Mean TV % | p95 TV % | Max TV % | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Typed decisions, full | 400 | 2,000 | 12 | 0.600000 | 0.488583 | 1.470000 | 10.160000 | [NPU](typed-decisions-full-npu.json) · [CPU](typed-decisions-full-reference.json) · [inputs](typed-decisions-full-suite.json) · [log](typed-decisions-full-log.txt) |
| Supplementary probes | 10 | 15 | 0 | 0.000000 | 0.220478 | 0.894986 | 0.953287 | [NPU](supplementary-development-npu.json) · [CPU](supplementary-development-reference.json) · [inputs](supplementary-development-suite.json) · [log](supplementary-development-log.txt) |
| Long input | 16 | 80 | 0 | 0.000000 | 0.484824 | 1.026097 | 5.495011 | [NPU](long-input-npu.json) · [CPU](long-input-reference.json) · [inputs](long-input-suite.json) · [log](long-input-log.txt) |
| BANKING77 card8 | 320 | 320 | 1 | 0.312500 | 0.426522 | 1.748165 | 6.640000 | [NPU](banking77-card8-npu.json) · [CPU](banking77-card8-reference.json) · [inputs](banking77-card8-suite.json) · [log](banking77-card8-log.txt) |
| CLINC150 domain10 | 300 | 300 | 1 | 0.333333 | 0.622212 | 1.741816 | 7.150715 | [NPU](clinc150-domain10-npu.json) · [CPU](clinc150-domain10-reference.json) · [inputs](clinc150-domain10-suite.json) · [log](clinc150-domain10-log.txt) |
| MASSIVE en-US | 180 | 180 | 0 | 0.000000 | 0.410759 | 1.398199 | 2.390000 | [NPU](massive-en-US-npu.json) · [CPU](massive-en-US-reference.json) · [inputs](massive-en-US-suite.json) · [log](massive-en-US-log.txt) |
| MASSIVE zh-TW | 180 | 180 | 1 | 0.555556 | 0.520536 | 1.750666 | 5.421084 | [NPU](massive-zh-TW-npu.json) · [CPU](massive-zh-TW-reference.json) · [inputs](massive-zh-TW-suite.json) · [log](massive-zh-TW-log.txt) |
| MASSIVE zh-CN | 180 | 180 | 0 | 0.000000 | 0.467484 | 1.785972 | 2.933054 | [NPU](massive-zh-CN-npu.json) · [CPU](massive-zh-CN-reference.json) · [inputs](massive-zh-CN-suite.json) · [log](massive-zh-CN-log.txt) |
| Legacy probes and requests | 7 | 8 | 0 | 0.000000 | 0.208718 | 0.723000 | 0.870000 | [NPU](legacy-npu.json) · [CPU](legacy-reference.json) · [inputs](legacy-suite.json) · [log](legacy-log.txt) |

## Overlapping historical subsets

These rows reuse decisions from the full typed and legacy reports. They are
additional mandatory gates, not extra independent requests or decisions.

| Suite | Requests | Decisions | Mismatches | Mismatch % | Mean TV % | p95 TV % | Max TV % | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Typed historical heldout | 368 | 1,840 | 11 | 0.597826 | 0.488395 | 1.471282 | 10.160000 | [Parent NPU](typed-decisions-full-npu.json) · [selection](qualification-plan.json) |
| Typed development | 16 | 80 | 0 | 0.000000 | 0.577013 | 1.451249 | 4.570457 | [Parent NPU](typed-decisions-full-npu.json) · [selection](qualification-plan.json) |
| Legacy accuracy probes | 4 | 4 | 0 | 0.000000 | 0.332500 | 0.807000 | 0.870000 | [Parent NPU](legacy-npu.json) · [selection](qualification-plan.json) |
| Legacy service requests | 3 | 4 | 0 | 0.000000 | 0.084937 | 0.204463 | 0.230000 | [Parent NPU](legacy-npu.json) · [selection](qualification-plan.json) |

The typed historical heldout selection contains source rows with
`index % 25 not in (0, 1)`; typed development contains `1, 26, …, 376`.
Legacy accuracy uses requests 0–3 and legacy service uses requests 4–6, with four
decisions in each selection. Exact index lists are frozen in the plan.

## What the 1% result means

The largest suite mean TV is **0.622212%** (CLINC150), and the largest decision
mismatch rate is **0.600000%** (full typed decisions). Individual errors can be
larger: the highest suite p95 TV is **1.785972%** (MASSIVE zh-CN), and the largest
individual TV is **10.160000%** (typed decisions, also present in its historical
heldout subset). This result does **not** assert that every example or every
probability differs by less than one percentage point.

TV is half the sum of absolute differences between the normalized original and
candidate probability distributions. The benchmark uses the existing API's
rounded probabilities. Decision mismatch retains the historical rules: published
choice labels, `noul > 0.5`, and rounded numeric scores. Argmax disagreement and
score errors are separately available in the raw reports. The p95 calculation
uses linear interpolation between sorted per-decision TV values. The unchanged
[metric implementation](../../benchmark_fidelity.py) defines these conventions.

This is **historical regression qualification**, not a fresh independent heldout
generalization claim or an assessment of task correctness against human labels.
These corpora were previously used in the project. MASSIVE translations share
semantic examples; synthetic long inputs derive from typed source rows. The
selection historically named heldout remains historical evidence. The independent
audit is independent recomputation of evidence, not a new dataset.

## Input and runtime integrity

Every request uses the same original CPU input, token IDs/hash, marker positions,
and reported token budget. The original **1024-token capacity** is retained with
768/1024 execution buckets; no shorter candidate truncation is substituted.
The original CPU source, checkpoint, tokenizer and configuration identities are
bound to the plan. The converted typed reference is independently checked against
the [original CPU JSONL](../qualification-2026-09-29/reference-wsl.jsonl).

The [frozen manifest](candidate-manifest.json) binds the exact corrected 768/1024
graphs and QNN backend identity. Actual encoder execution is verified per request
and in aggregate. All nine runs report zero CPU fallbacks, cache misses, cache
errors and cache writes. Their context paths and backend fingerprints match
[pre-qualification cache evidence](context-cache-before-run.json), which records
both prepared cache byte hashes and successful reloads before the plan froze.
CPU preprocessing and output heads remain part of the normal service; zero CPU
EP fallback here specifically describes the accelerated encoder.

The qualification plan SHA256 is
`459f4c82420c4abab534e910cf0c1a5a95f852e0d8084a328fcd1fb1b3e03942`.
Exact file hashes for every suite, CPU reference, NPU report, graph and qualified
harness are available in the plan and audits. The local review also checked
byte-identical copies of all 40 remote evidence files and all six current
qualified harness files.

## Reproduce the offline audit

From the repository root, with the accompanying historical references intact:

```bash
python qualify_one_percent.py audit \
  --output-dir reports/qualification-2026-10-03 \
  --audit-output reports/qualification-2026-10-03/recomputed-audit.json
```

Use a new output filename each time. This command needs no model loading or NPU.
Keep the directory-local [.gitattributes](.gitattributes): frozen files are
preserved byte-for-byte, including the historical CRLF supplementary suite.
The [format provenance](../evaluation-plans/fidelity-probes-format-provenance-2026-10-03.json)
documents its identity with the old Git LF representation. The
[qualification guide](../../docs/one-percent-qualification.md) explains the
inventory and source checks.

The separate [HTTP/systemd service acceptance](../precision-service-2026-10-03/README.md)
also passed: all ten supplementary requests reproduced the qualified NPU outputs,
with zero CPU fallback or restarts and a 3,858,608,128-byte cgroup memory peak
under the 4 GiB limit. The service returned to its original inactive state while
retaining the selected manifest on disk. This finite integration workload is
additional deployment evidence, not an extension of the fidelity scope above.
