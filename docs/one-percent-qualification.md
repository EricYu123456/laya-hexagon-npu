# One-percent historical fidelity qualification

The [2026-10-03 completed run](../reports/qualification-2026-10-03/README.md)
passes all nine suites and four required overlapping subsets, independently
recomputed on the Pi and in WSL. Its [separate service acceptance](../reports/precision-service-2026-10-03/deployment.json)
also passes exact HTTP replay and the existing 4 GiB memory limit.

`qualify_one_percent.py` fixes the complete historical input inventory before
running an improved candidate. Every suite must have **decision mismatch at most
1% and mean total variation at most 0.01**, identical original token IDs/markers,
and actual QNN HTP encoder execution with zero CPU EP fallback. Existing 5% pass
flags are not used for this decision. Probabilities and decision differences are
recomputed from raw original and candidate answers.

| Suite | Requests | Decisions | Maximum mismatches at 1% |
| --- | ---: | ---: | ---: |
| Complete typed-decisions public split | 400 | 2,000 | 20 |
| Supplementary multilingual/long/switching probes | 10 | 15 | 0 |
| Synthetic 1024-token validation | 16 | 80 | 0 |
| BANKING77, eight fixed intents | 320 | 320 | 3 |
| CLINC150, ten domains | 300 | 300 | 3 |
| MASSIVE en-US | 180 | 180 | 1 |
| MASSIVE zh-TW | 180 | 180 | 1 |
| MASSIVE zh-CN | 180 | 180 | 1 |
| Legacy accuracy probes and service requests | 7 | 8 | 0 |
| Total executed | 1,593 | 3,263 | Every suite must pass separately |

The auditor additionally gates the historical typed held-out subset (368 requests,
1,840 decisions, at most 18 mismatches), fixed typed development subset (16/80,
zero mismatches), four legacy accuracy decisions, and four legacy service
decisions. These are overlapping subsets, not additional independent examples.
The complete typed split covers the earlier `benchmark_typed_decisions.py` corpus.
The legacy suite preserves the four inputs from `accuracy-probes.json`, plus the
exact positive, negative and score requests from `verify.py`/`example.json`.
It is compared against fresh **unchanged, untruncated original CPU Laya**, never
the old 64-token/clamped output artifacts.

The [immutable inventory](../reports/evaluation-plans/one-percent-historical-inventory.json)
records the exact historical suite and original-reference file hashes. Its hash
is also pinned in the driver, so input removal or replacement cannot silently
create an easier evaluation. Model diagnostics on random tensors in `npu/test_*`
are numerical implementation tests, not additional language datasets. API 422
tests, cache replay, and service integration remain separate deployment checks.

## Capture the missing original reference

The existing typed, long, supplementary and five external original CPU references
are reused byte-for-byte. Only the combined legacy suite needs a new pristine CPU
reference. This can run in the configured WSL build environment:

```bash
python benchmark_probes.py --suite tests/legacy-fidelity-probes.json \
  --backend cpu --threads 4 --output .work/legacy-reference.json
```

The driver requires identical upstream Laya source hashes, package version,
checkpoint, tokenizer and configuration across all references and candidates.
It losslessly converts the 400-case frozen typed JSONL reference into probe
format, retaining original answers, timings, input hashes and token metadata.
This lets every typed request also obtain the probe harness's per-request NPU
and collated-bucket counter checks. During every offline audit, the converted
inputs, order, original answers, token metadata, timings and runtime identity
are independently checked against the byte-pinned original CPU JSONL. Updating
a derived reference's reported hash cannot substitute different CPU answers.

## Freeze and execute one candidate

Choose the final candidate using only declared development inputs. Prepare its
corrected 768 and 1024 graphs and compiled contexts. On the Pi, after stopping
other NPU jobs and sourcing the target environment:

```bash
source .venv/bin/activate
source npu/env.sh
python qualify_one_percent.py prepare \
  --output-dir reports/one-percent-candidate \
  --dataset-path .work/typed-decisions.parquet \
  --legacy-reference .work/legacy-reference.json \
  --manifest npu/fidelity/manifest.json
python qualify_one_percent.py run --output-dir reports/one-percent-candidate
```

`prepare` requires a new output directory and freezes all input/reference hashes,
candidate graph and manifest hashes, checkpoint identity, harness source hashes,
and every required subset before inference. It verifies the pinned original
typed parquet checksum and copies the exact manifest bytes to
`candidate-manifest.json` beside the plan. Every candidate's embedded manifest
must equal that frozen snapshot, including the complete offset-correction
metadata; an unchanged claimed manifest hash cannot hide a removed correction
block. Finish source changes and synchronization before `prepare`, because
changing the inference/audit harness invalidates the run's frozen source hashes.
Each suite runs sequentially in a fresh process using the frozen manifest. Logs
and full candidate records are kept beside the plan. One numerical failure does
not skip the other suites.

If interrupted, use the same graph/harness and `run --resume`; complete existing
reports are audited before reuse. A different candidate needs a new directory.
Reports are never overwritten. If an earlier audit file already exists, use
`--audit-output <new-path>` on the resumed run.

Offline verification requires no model loading or NPU:

```bash
python qualify_one_percent.py audit \
  --output-dir reports/one-percent-candidate \
  --audit-output reports/one-percent-candidate/recomputed-audit.json
```

Exit zero and `passed: true` require every suite and subset to pass, with no
missing evidence. The auditor validates exact coverage, probabilities, markers,
input usage, original source/checkpoint identity, graph hashes, offset-correction
backend binding, and per-request plus aggregate NPU counters. A report from CPU
QDQ diagnostic execution cannot qualify. The threshold applies to aggregate
decision mismatch and mean TV; it does not promise every probability differs by
less than one percentage point.

The qualification directory can be copied intact to another host for this
offline audit. Keep the committed historical inventory, original typed CPU
JSONL, and audit source in the accompanying repository checkout. Offline auditing
uses the copied manifest snapshot and does not need to resolve its original Pi
graph paths. Preserve file bytes when copying: `.gitattributes` normally pins
report JSON and JSONL to LF even with Windows `core.autocrlf=true`. Specific
`-text` overrides preserve the frozen historical inventory and legacy probe
suite's original CRLF bytes. The `tests/fidelity-probes.json -text` override
also preserves that
supplementary suite's historical CRLF bytes, whose SHA256 is pinned in its
original CPU reference and the immutable inventory. Its previous Git LF form
has identical JSON and all ten request inputs; the
[format provenance](../reports/evaluation-plans/fidelity-probes-format-provenance-2026-10-03.json)
records both byte hashes and the independent semantic comparison. Do not
normalize this file or rewrite its original reference to satisfy a byte check.
The frozen qualification directory also contains this original CRLF suite as
`supplementary-development-suite.json`. Before adding that directory to Git,
place a directory-local `.gitattributes` containing `* -text` inside it; this
overrides repository report normalization and preserves every frozen evidence
file byte-for-byte when publishing or checking out the report. Audit again after
a fresh checkout to verify that its recorded hashes remain intact.
Original host paths and runtime metadata remain provenance; cross-host CPU/NPU
latencies are not a same-host speed comparison.

## Evaluation reuse and independent claims

All these corpora have already been evaluated in the project. They are historical
regression tests, including the selection historically named "held-out". Do not
fit calibration ranges, correction tables, temperatures, lookup overrides or
model selection to their current evaluation errors. If these results influence
further tuning, disclose that reuse. A fresh independent generalization claim
needs a separately declared, untouched dataset after the model recipe is frozen.
MASSIVE translations share 180 semantic IDs, and synthetic long inputs derive
from typed source rows; neither should be counted as independent natural corpora.
