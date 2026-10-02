# Precision development evidence — 2026-10-03

This bundle records the earlier 768-token development stage. The complete
768/1024 recipe subsequently **passed all nine historical NPU suites and four
declared subsets**; see the [qualification evidence](../qualification-2026-10-03/)
and [audit](../qualification-2026-10-03/one-percent-audit.json). Across those
gates, the highest decision mismatch rate was 0.6% and highest mean TV was
0.622212%. The development measurements below remain unchanged and are separate
from that full historical qualification.

These are actual Rubik Pi 3 NPU runs on the same 16 reserved typed-decisions
cases (80 decisions), using four CPU threads, one warm-up case, cached QNN
contexts, and zero CPU encoder fallbacks. The input, checkpoint, tokenizer,
original FP32 reference answers and token identities match across all three runs.

| 768-token model | Decision mismatches | Mean TV | Mean case latency | Corrected ONNX size |
|---|---:|---:|---:|---:|
| September 29 baseline | 4/80 | 2.4569% | 4.764 s | 111.36 MiB |
| Two-term Conv weights | 2/80 | 2.5244% | 6.108 s | 220.70 MiB |
| Feature bands + weights + affine refinement | 0/80 | 0.5770% | 8.782 s | 225.25 MiB |

The combined candidate reduces mean total variation (TV) by 76.5%
relative to the baseline, with 0/80 decision mismatches. Its mean TV of
0.5770% is below 1% for this development selection. This is a
combined-recipe result: two-term weights alone did not improve mean TV. The
candidate's TV p95 is 1.4512% and maximum is 4.5705%;
there is no per-example 1% error bound. The original reports retain their older
5% pass criteria; the 1% comparison here uses their recorded counts and mean TV.

The recipe separates CLS and non-CLS residuals within feature bands of
631, 127, 6 and 4 channels, derived from the reserved calibration selection
using amplitude thresholds 100, 500 and 2000. It folds compatible LayerNorm
affines, represents projection weights using two per-channel INT8 Conv terms,
and gives the remaining three learned LayerNorm affines U16 elementwise
arithmetic. Existing attention MatMul refinement and the original input
contract remain. Scaled LayerNorm introduces an epsilon approximation, checked
at FP32 export before quantization. All 1,034 Convs receive fresh graph-specific
zero-input HTP offset calibration with three original-spatial-shape controls.

Mean benchmark case latency increases **84.3%**
(1.84×), and the corrected graph is 2.02× as
large. These are end-to-end case timings, not isolated kernel timings. The
FP32 reference ran on a different host/thread configuration and cannot support
a CPU-versus-NPU speed comparison. CPU-only development experiments are not
included as NPU measurements.

The offline compile-and-verified-reload command completed successfully in
21 min 25.39 s with 6.54 GiB peak process RSS. First context preparation took
1191.82 s; verified cached reload took 9.75 s. This memory measurement covers
the compilation process, not steady inference. The raw log preserves earlier
capability warnings and the final successful completion, reload and exit status.

Evidence files are byte-for-byte copies with SHA256 values in
[file-inventory.json](file-inventory.json):

- Actual NPU reports and matching per-case records:
  [baseline](baseline-dev.json) / [cases](baseline-dev.cases.jsonl),
  [weights](weight16-npu-dev.json) / [cases](weight16-npu-dev.cases.jsonl),
  [combined candidate](features32-npu-dev.json) / [cases](features32-npu-dev.cases.jsonl).
- [Unchanged original FP32 reference cache](reference-wsl.jsonl), SHA256
  `112c482ae5b2f7b88dfb2784375409484a115d75b39d3e08f671f20bd013233e`.
- [Compile summary](feature768-compile-summary.json) and
  [raw pipeline log](features32_768_pipeline.txt). The `.txt` rename preserves
  all bytes; the summary's original `.log` filename is mapped in the inventory.

Original machine paths in the copied reports are retained as provenance.
Corrected graph sizes were recorded from the exact Pi graph paths identified
by the reports; the large model/context files are omitted from this bundle.
The combined corrected 768 graph SHA256 is
`a3ba4ac988f726d61d06543e723612e63c3047fe53d9133b8a7b5858a7365003`.

The [qualified manifest](../qualification-2026-10-03/candidate-manifest.json)
identifies the corrected graphs stored on the Pi under
`npu/fidelity/precision-2026-10-03/{768,1024}`. Qualification checked each
declared suite and subset separately under strict NPU execution with matching
input identities. This is historical regression evidence; it does not imply a
per-example 1% bound or new independent held-out generalization.
