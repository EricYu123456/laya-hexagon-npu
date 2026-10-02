# Deploy the verified NPU graph set

Run on the Pi as `ubuntu` in `/home/ubuntu/laya-service`. The Python environment,
original multilingual checkpoint, and corrected 768/1024 graphs must already be
installed. Keep graphs and their correction provenance together in stable
directories; generated binaries are not distributed in Git.

The `precision-2026-10-03` graph set has passed the [complete one-percent
historical fidelity audit](../reports/qualification-2026-10-03/one-percent-audit.json)
and fresh [service acceptance](../reports/precision-service-2026-10-03/deployment.json).
The selected manifest is installed as `npu/fidelity/manifest.json`; its SHA256 is
`159e6f8cab13b5362c6241b27e2be1bdfc0a7d402b9521205441a50c9362951b`.
Acceptance temporarily started the service, then restored its original
**inactive/dead** state. The instructions below start it when needed.

## Prepare and verify before switching the service

Merge the two corrected single-bucket manifests into a **new** file in
`npu/fidelity/`, using their actual paths. The merge tool validates graph hashes,
checkpoint/input policies, and correction backend compatibility. It refuses to
overwrite an existing manifest. Stop the service and finish other NPU work before
preparing contexts so compilation has exclusive accelerator access and memory.

```bash
systemctl --user stop laya.service
.venv/bin/python -m npu.merge_manifests --output npu/fidelity/manifest.next.json \
  npu/fidelity/precision-2026-10-03/768/manifest.json \
  npu/fidelity/precision-2026-10-03/1024/manifest.json
source npu/env.sh
export LAYA_NPU_MANIFEST="$PWD/npu/fidelity/manifest.next.json"
export LAYA_NPU_CONTEXT_CACHE=1
.venv/bin/python npu/compile_contexts.py \
  --manifest "$LAYA_NPU_MANIFEST" --verify-reload
```

Run precompilation outside the service cgroup. The measured precision-graph
compile-and-reload commands peaked at 6.54 GiB for 768 and 6.59 GiB for 1024,
above the unit's 4 GiB limit. The service and CLI must use the
same user, QNN environment, provider options, and cache directory. Cache keys bind
graph and QNN binary hashes, so relocating identical graphs does not require
recompilation.

The [1024 preparation](../reports/precision-build-2026-10-03/README.md) used two
temporary 8 GiB swap files during its approximately 83-minute compile-and-reload
command. Both were removed before qualification; no peak swap usage is claimed.
Do not change `fstab` for this offline step. After the compiler exits, remove
temporary swap and verify cached loading before service testing. The measured
cached reloads were 9.75 s for 768 and 12.59 s for 1024. Compilation process RSS,
configured swap and steady service cgroup memory are separate measurements.

Before activation, require a successful [complete nine-suite audit](one-percent-qualification.md)
for the exact corrected graph hashes, including every required subset and the
original 1024/256-token budgets. Each suite requires decision mismatch <=1%, mean
TV <=0.01, identical inputs and strict HTP execution. The 15-decision supplementary
suite is a required zero-mismatch gate; it is not an exception to acceptance.
Capture standalone supplementary outputs from that same graph set for HTTP replay:

```bash
.venv/bin/python benchmark_probes.py \
  --reference reports/development-2026-09-28/probe-reference-wsl.json \
  --output .work/deployment-probes.json --threads 4
```

Use a fresh report path on each run and preserve it for exact HTTP replay. The
qualified precision graph has [0/15 supplementary decision differences and
0.2205% mean TV](../reports/qualification-2026-10-03/supplementary-development-npu.json).
The low-level probe report's legacy pass flag does not replace the one-percent
auditor. The September 29 graph's failed supplementary result remains historical
evidence and is not an activation justification for this graph set.

For these exact qualified graph hashes, the frozen supplementary suite/reference
and NPU outputs can be reused for replay as shown below. A changed candidate needs
its own matching outputs and complete qualification.

## Activate and verify HTTP behavior

Back up the existing default manifest and installed user unit before replacing
them. Keep the new and default manifests in the same directory so relative graph
paths remain valid. After activation, the default manifest is
`npu/fidelity/manifest.json`:

```bash
mkdir -p .work ~/.config/systemd/user
if test -f npu/fidelity/manifest.json; then
  cp -p npu/fidelity/manifest.json ".work/manifest.before-$(date +%s).json"
fi
if test -f ~/.config/systemd/user/laya.service; then
  cp -p ~/.config/systemd/user/laya.service ".work/laya.service.before-$(date +%s)"
fi
mv npu/fidelity/manifest.next.json npu/fidelity/manifest.json
export LAYA_NPU_MANIFEST="$PWD/npu/fidelity/manifest.json"
install -m 644 systemd/laya.service ~/.config/systemd/user/laya.service
systemctl --user daemon-reload
systemctl --user restart laya.service
.venv/bin/python verify_service.py \
  --suite reports/qualification-2026-10-03/supplementary-development-suite.json \
  --reference reports/qualification-2026-10-03/supplementary-development-reference.json \
  --expected-npu reports/qualification-2026-10-03/supplementary-development-npu.json \
  --output .work/http-verification.json
```

The unit starts `systemd/run-laya.sh`, which loads the matching QNN environment.
`service.env` selects `LAYA_DEVICE=npu` and four CPU threads. Readiness requires
actual strict NPU warmup and a graph set covering the original input capacity.
The HTTP verifier checks invalid-input rejection, exact replay of standalone
NPU outputs, token usage, per-request NPU counters, both bucket transitions, and
zero CPU fallbacks or cache misses. Its requests fit the API's four-question,
16,000-character limits; use the CLI for five-question benchmark cases.
`integration_passed` (also exposed as `passed`) describes those integration checks;
`development_fidelity_passed` independently reports the verifier's supplementary
fidelity gate. Neither flag substitutes for the complete one-percent audit.

Record `MainPID`, `InvocationID`, `NRestarts`, `MemoryMax`, `MemoryPeak`, and
`ControlGroup` with `systemctl --user show laya.service` before and after the
workload. In `/sys/fs/cgroup/<ControlGroup>/`, retain `memory.max`, `memory.peak`,
and `memory.events`. Require the same invocation, no restart/OOM, and a peak below
the 4 GiB limit. Process RSS alone is not this cgroup memory check.

The October 3 [HTTP check](../reports/precision-service-2026-10-03/service-http.json)
passed exact replay of 10 requests / 15 decisions, invalid-input checks and both
bucket transitions. The same service PID/invocation survived the workload with
zero restarts, OOMs or memory-limit events. Its cgroup peak was **3,858,608,128 bytes
(3.593609 GiB)** below the 4 GiB limit. Strict NPU execution had zero CPU fallbacks
and no cache misses, errors or writes. The recorded before/after swap snapshots
were zero and both temporary build swap files were absent; no peak swap usage is
claimed. [Final state](../reports/precision-service-2026-10-03/service-final.json)
confirms the service was stopped after acceptance while the selected manifest
remained installed.

Changing the checkpoint, graph, ORT/QNN binaries, or quantization recipe requires
new correction/cache preparation and fidelity evaluation. To roll back, restore
the backed-up compatible manifest and unit, reload systemd, and restart. Starting
or restarting does not change whether the unit is enabled at login.
