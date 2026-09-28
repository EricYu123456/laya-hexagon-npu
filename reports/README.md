# Recorded fidelity experiments

[development-2026-09-28](development-2026-09-28) contains development evidence from 16 fixed cases and 80 decisions. The corrected `refined32` candidate reaches the development thresholds. These are not held-out results or deployment qualification.

| Report | Provider | Decision mismatch | Mean total variation |
| --- | --- | ---: | ---: |
| [grouped16-npu.json](development-2026-09-28/grouped16-npu.json) | Actual QNN HTP V68 | 12.50% | 8.5510% |
| [grouped16-cpu-diagnostic.json](development-2026-09-28/grouped16-cpu-diagnostic.json) | CPUExecutionProvider | 7.50% | 5.7729% |
| [clsconv16-npu.json](development-2026-09-28/clsconv16-npu.json) | Actual QNN HTP V68 | 20.00% | 18.3522% |
| [clsconv16-cpu-diagnostic.json](development-2026-09-28/clsconv16-cpu-diagnostic.json) | CPUExecutionProvider | 2.50% | 2.8554% |
| [refined32-npu-before-offset-correction.json](development-2026-09-28/refined32-npu-before-offset-correction.json) | Actual QNN HTP V68 | 36.25% | 27.1874% |
| [refined32-corrected-npu.json](development-2026-09-28/refined32-corrected-npu.json) | Actual QNN HTP V68 | **5.00%** | **2.4568%** |

Each report has a sibling `.cases.jsonl` containing original/candidate answers, input identities, timings, and per-question errors. `reference-pi.jsonl` preserves the original Pi CPU reference. `reference-portability.json` compares original Pi CPU and WSL CPU outputs; it does not compare their speed. Build/export sidecars record graph configuration and validation.

Metadata retains the original hosts, package versions, source hashes, and absolute paths. Artifact hashes identify the models actually measured; model binaries and context caches are not included. The corrected development run used a precompiled context with zero CPU fallbacks. See [the evaluation method](../docs/fidelity-method.md) for the fixed data split, metric definitions, reproduction commands, and pending acceptance work. The [held-out selection plan](evaluation-plans/typed-decisions-heldout.json) references the committed exclusion list from before full evaluation.
