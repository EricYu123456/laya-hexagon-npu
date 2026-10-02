# Temporary service acceptance — 2026-10-03

**PASS:** the selected one-percent-qualified model reproduced every standalone
NPU answer, usage value and model identity through the HTTP service. All ten
supplementary requests (15 decisions) passed, with **zero decision mismatches**
and **mean TV 0.220478%** against the unchanged CPU reference. Five invalid
requests returned HTTP 422 without invoking the NPU. Both 768→1024 and
1024→768 transitions were exercised.

The run started at `2026-10-02T19:43:19.292633+00:00` and completed at
`2026-10-02T19:44:44.814442+00:00` (2026-10-03 in Asia/Taipei). The service was
initially **inactive/dead**, ran temporarily, and finished **inactive/dead with
PID 0**. The qualified manifest remains selected on disk. The acceptance script
did not enable the service or leave it running.

## Execution and memory

| Check | Verified result |
| --- | --- |
| Replay encoder calls | 15: bucket 768 × 11, bucket 1024 × 4 |
| Startup warmup | One additional bucket-768 call; 16 total calls |
| CPU EP fallback | Zero |
| Prepared context hits | Five including startup |
| Context misses/errors | Zero |
| Process continuity | PID 428871 and invocation `10a14be834b540f2989e304be57510d8` unchanged during replay |
| Service restarts | Zero |
| Cgroup memory peak | **3,858,608,128 bytes (3.593609 GiB)** |
| Cgroup memory limit | 4,294,967,296 bytes (4 GiB) |
| Margin below limit in this run | 436,359,168 bytes (416.144531 MiB) |
| Memory limit/OOM events | Zero in recorded cgroup counters |
| Observed swap | `memory.swap.current = 0` at all running snapshots; `/proc/swaps` had no entries at recorded checks |
| Temporary build swap files | Both required absent before activation and after replay |
| Final state | Inactive/dead, selected manifest retained, no cleanup errors |

This is a **finite acceptance workload**, not a sustained-load, concurrency,
long-duration reliability or memory-headroom guarantee. Swap values are recorded
observations, not a lifetime `memory.swap.peak` measurement. Broader fidelity is
established separately by the [nine-suite qualification](../qualification-2026-10-03/README.md).

## Evidence and independent checks

The [deployment report](deployment.json) binds the exact qualification plan,
independent audit, graph/manifest/cache hashes, service configuration and script.
Its [fresh qualification recomputation](qualification-recomputed.json) agrees
with the [copied independent audit](independent-audit.json) for all nine suites
and four overlapping subsets. The plan and original qualification evidence are
available [here](../qualification-2026-10-03/qualification-plan.json).

The [HTTP report](service-http.json) contains all responses, independently
recomputed decision metrics, usage and accelerator deltas, invalid-request
checks and health observations. It reproduces the exact
[qualified supplementary NPU report](../qualification-2026-10-03/supplementary-development-npu.json)
using its [frozen inputs](../qualification-2026-10-03/supplementary-development-suite.json)
and [original CPU reference](../qualification-2026-10-03/supplementary-development-reference.json).
Its historical `development_fidelity_passed` flag still means 5%; the acceptance
script separately enforces the strict 1% supplementary thresholds.

State evidence: [initial](service-initial.json),
[before replay](service-before-replay.json), [after replay](service-after-replay.json),
[before stop](service-before-stop.json), and [final](service-final.json).
Logs: [execution](execution-log.txt), [HTTP replay](service-http-log.txt), and
[systemd journal](service-journal.txt). Configuration: [environment](service.env)
and [installed unit](laya.service). The [old manifest backup](manifest.before.json)
was preserved. The [prepared-context evidence](context-cache-before-run.json)
records the cache bytes before qualification; both installed copies match it.

An independent local read-only review checked all 16 fetched remote raw-file
hashes, exact HTTP replay and metric recomputation, the copied qualification
identities, cache provenance, original/final state, and cgroup/process continuity.
The [acceptance script](acceptance-script.txt) is an exact byte copy of the script
recorded in `deployment.json`; executing it on the target host changes the default
manifest and temporarily starts the service. The `.txt` copy is retained as evidence.

## Artifact identities and byte preservation

| Artifact | SHA256 |
| --- | --- |
| Selected/default manifest after success | `159e6f8cab13b5362c6241b27e2be1bdfc0a7d402b9521205441a50c9362951b` |
| Qualification plan | `459f4c82420c4abab534e910cf0c1a5a95f852e0d8084a328fcd1fb1b3e03942` |
| Independent qualification audit | `2eb39bbd801edad6be44ad7aa3802c491131bf498b16c70230c0639b450b826c` |
| Deployment report | `c052f656d0e2a5d1b49f3ec2743ec8da733de436134e2b0d718ac821ecbbe730` |
| HTTP report | `9bbfc067310170ee7f26990bd342504128e66cb3049805da5d7ee66685572214` |
| Acceptance script | `c05d67dcc1ad982f7ec8970721b924657dc9f4126678cacc372408255c410720` |

[file-inventory.json](file-inventory.json) records file sizes and SHA256s for the
published bundle, excluding itself and the ignored original `service-http.log`.
The published `service-http-log.txt` preserves that log's exact bytes. Keep the
directory-local [.gitattributes](.gitattributes) to prevent Git newline conversion.
Recorded host paths are provenance; the bundle links and inventory paths are
relative so the evidence can be inspected after relocation.
