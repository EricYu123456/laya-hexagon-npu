# Precision build evidence — 2026-10-03

This bundle records the earlier context-build stage. The complete 768/1024
recipe subsequently **passed all nine historical NPU suites and four declared
subsets**; see the [qualification evidence](../qualification-2026-10-03/) and
[audit](../qualification-2026-10-03/one-percent-audit.json). The highest decision
mismatch rate was 0.6% and highest mean TV was 0.622212%. Compilation evidence
below remains distinct from those later fidelity results.

The strict HTP 1024-token context compiled successfully and reloaded from its
saved cache, with zero CPU encoder fallbacks and command exit status 0.

| Completed build measurement | Value |
|---|---:|
| Compile-and-reload command elapsed time | 1 h 22 min 44 s |
| First context preparation | 4737.72 s |
| Verified cached reload | 12.59 s |
| Peak process RSS reported by GNU time | 6.5901 GiB |
| Corrected ONNX graph | 236,302,473 bytes |
| Saved 1024 context before qualification | 409,409,498 bytes |

Two temporary 8 GiB swap files (16 GiB configured total) were used during the
build, according to the operator session observation. Both were removed before
qualification. The preserved queue script places their cleanup before cache
recording and qualification startup; a read-only device observation at
2026-10-02T17:52:56Z confirms zero configured swap and an empty `/proc/swaps`
during qualification. No peak process or system swap measurement is claimed.
The GNU time `Swaps: 0` counter is not a peak-memory measurement. Process RSS,
configured swap capacity and steady inference memory are distinct quantities.

- [Compilation summary and all ordered compiler stages](1024compile-summary.json).
- [Original pipeline log, renamed without changing bytes](features32_1024_pipeline.txt).
- [Hashes and sizes of both saved contexts before qualification](one-percent-cache-before-qualification.json).
- [Read-only graph-size and memory observation](feature1024-graph-memory-observation.txt).
- [Executed queue script snapshot](queue_full_qualification.txt), preserved as text for evidence, not as an instruction to rerun it.
- [SHA256 file inventory](file-inventory.json).

The [completed 768-token development bundle](../precision-development-2026-10-03/README.md)
contains its separate compilation and measured fidelity evidence. Large ONNX
and context binaries are omitted from both evidence bundles.

The original `1024compile-summary.json` retains its build-time `PENDING`
qualification status as an unchanged historical snapshot. Current qualification
status is recorded in the linked audit. Its [manifest](../qualification-2026-10-03/candidate-manifest.json)
identifies the corrected Pi graphs under
`npu/fidelity/precision-2026-10-03/{768,1024}`. Compilation RSS and configured swap
do not establish service memory requirements.
