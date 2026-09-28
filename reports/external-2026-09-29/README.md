# External dataset fidelity evaluation

These are new, fixed **adapted choice tasks** for comparing the deployed Hexagon
NPU encoder with unchanged original Laya. They were not used for our quantization
calibration or model selection. We cannot establish whether original Laya's
pretraining contained these public corpora. No graph or calibration changes are
allowed during this evaluation.

| Suite | Fixed selection | Decisions |
| --- | --- | ---: |
| BANKING77 card/payment support | All official test rows from eight fixed intents | 320 |
| CLINC150 domain routing | Two test rows per each of 150 intents, mapped to the ten official domains | 300 |
| MASSIVE English | Ten test IDs per each of eighteen scenarios | 180 |
| MASSIVE Traditional Chinese | The same selected IDs, `zh-TW` | 180 |
| MASSIVE Simplified Chinese | The same selected IDs, `zh-CN` | 180 |

MASSIVE has 180 paired semantic examples across three languages, not 540
independent semantic examples. Each language is reported separately. The selected
BANKING77 intents are payment fees, unrecognized payments, wrong exchange rates,
declined payments, pending payments, refund requests, reverted payments, and
duplicate charges. CLINC150's out-of-scope partition is not used: an unsupported
intent can still belong to a supported domain, so it is not a clean extra domain.

The [frozen plan](../evaluation-plans/external-datasets-2026-09-29.json) records
selection rules, all suite hashes, unchanged graph/checkpoint hashes and the
original thresholds: decision differences ≤5%, mean total variation ≤0.05, and
exact input identity. Every suite must pass separately. The aggregate cannot hide
a failed dataset or language. Per-intent diagnostics have small denominators and
are not additional primary gates. Gold-label accuracy is auxiliary; these are
not native BANKING77 77-way, CLINC150 150-way, or MASSIVE 60-intent leaderboard
results.

Every row in a suite has identical question wording and option order. Only the
raw source utterance is passed as `state`; labels and source IDs stay in metadata.
A tokenizer-only preflight verifies complete state, instruction and option text,
as well as markers. Merely retaining all markers would not catch semantically
truncated option descriptions. The [recorded preflight](tokenizer-preflight.json)
passes all 1,160 inputs, whose complete lengths range from 158 to 233 tokens.
MASSIVE uses all eighteen scenario choices through
the original CLI API; it is not a test of the HTTP service's twelve-option limit.

## Sources, attribution and licenses

- **BANKING77**, Casanueva et al., *Efficient Intent Detection with Dual Sentence
  Encoders* (2020): [official PolyAI dataset](https://github.com/PolyAI-LDN/task-specific-datasets),
  revision `57ec275d8078af65b7731c2a98be812d844a6d6b`. Source text and labels are
  licensed under [CC BY 4.0](banking-LICENSE.txt).
- **CLINC150**, Larson et al., *An Evaluation Dataset for Intent Classification
  and Out-of-Scope Prediction* (EMNLP 2019): [official repository](https://github.com/clinc/oos-eval),
  revision `828f8093932c8fe6ca7936c3d2e52903b1c523de`. Source text and labels are
  licensed under [CC BY 3.0](clinc-LICENSE.txt).
- **MASSIVE 1.1**, FitzGerald et al., *MASSIVE: A 1M-Example Multilingual Natural
  Language Understanding Dataset with 51 Typologically-Diverse Languages* (2022):
  [official Amazon project](https://github.com/alexa/massive) and its versioned
  1.1 archive, licensed under [CC BY 4.0](massive-LICENSE.txt).

The suites select source test utterances without rewriting them and add our
fixed questions, option descriptions, and evaluation metadata. Source data keeps
the licenses above; the repository's code license does not replace them.
[sources.json](sources.json) pins every downloaded file by SHA256. Raw source
archives are excluded from Git; only the selected adaptations are included.

## Reproduction

`prepare_external_benchmarks.py` verifies source checksums and deterministically
regenerates the suites. It can download missing sources as data without executing
remote dataset scripts. Use fresh output and plan paths when reproducing so the
original frozen files cannot be overwritten.

For each suite in the plan, use its `suite`, `reference`, and `candidate` paths:

```bash
python benchmark_probes.py --backend cpu --threads 4 \
  --suite <suite> --output <reference>
# On the Pi, using the already qualified manifest and precompiled contexts:
source npu/env.sh
export LAYA_NPU_MANIFEST="$PWD/npu/fidelity/manifest.json"
export LAYA_NPU_CONTEXT_CACHE=1
python benchmark_probes.py --backend npu --threads 4 \
  --suite <suite> --reference <reference> --output <candidate>
```

The original CPU runs use WSL; NPU runs use the physical Pi. These measurements
compare numerical fidelity, not hardware speedup. The existing probe harness
labels raw reports as supplementary; the separate frozen plan and offline
summary establish the scope of this external evaluation.
