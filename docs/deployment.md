# Deploy the verified NPU graph set

Run on the Pi as `ubuntu` in `/home/ubuntu/laya-service`. The Python environment,
original multilingual checkpoint, and corrected 768/1024 graphs must already be
installed. Keep graphs and their correction provenance together in stable
directories; generated binaries are not distributed in Git.

## Prepare and verify before switching the service

Merge the two corrected single-bucket manifests into a **new** file in
`npu/fidelity/`, using their actual paths. The merge tool validates graph hashes,
checkpoint/input policies, and correction backend compatibility. It refuses to
overwrite an existing manifest. Stop the service and finish other NPU work before
preparing contexts so compilation has exclusive accelerator access and memory.

```bash
systemctl --user stop laya.service
.venv/bin/python -m npu.merge_manifests --output npu/fidelity/manifest.next.json \
  npu/fidelity/qualified-2026-09-29/768/manifest.json \
  npu/fidelity/qualified-2026-09-29/1024/manifest.json
source npu/env.sh
export LAYA_NPU_MANIFEST="$PWD/npu/fidelity/manifest.next.json"
export LAYA_NPU_CONTEXT_CACHE=1
.venv/bin/python npu/compile_contexts.py \
  --manifest "$LAYA_NPU_MANIFEST" --verify-reload
```

Run precompilation outside the service cgroup. The measured 768 preparation
peaked at 5.04 GiB, above the unit's 4 GiB limit. The service and CLI must use the
same user, QNN environment, provider options, and cache directory. Cache keys bind
graph and QNN binary hashes, so relocating identical graphs does not require
recompilation.

The 1024 graph's first preparation exhausted the board's physical RAM during
cache materialization. It may need temporary swap for this offline step. Do not
change `fstab` for this purpose. After the compiler exits and releases its memory,
disable that temporary swap and verify a cache reload in a new process before
testing the service. Compilation memory and cached inference memory are separate
measurements.

Before activation, require the declared primary NPU fidelity checks to pass,
including the original 1024-token budget. Also capture and disclose the fixed
supplementary development probes:

```bash
.venv/bin/python benchmark_probes.py \
  --reference reports/development-2026-09-28/probe-reference-wsl.json \
  --output .work/deployment-probes.json --threads 4
```

Use a fresh report path on each run. These development probes supplement the
[declared fidelity evaluations](fidelity-method.md); they do not replace them.
The recorded 15-decision suite has one decision difference and exits with code 1;
its accuracy gate remains failed. Preserve that result for exact HTTP replay.
The deployed candidate passes the primary 1,840-decision and synthetic 80-decision
evaluations, with this supplementary limitation disclosed in the
[acceptance clarification](../reports/qualification-2026-09-29/service-acceptance.md).

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
  --reference reports/development-2026-09-28/probe-reference-wsl.json \
  --expected-npu .work/deployment-probes.json \
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
`development_fidelity_passed` independently reports the supplementary accuracy
gate and is false for the currently deployed candidate.

Record `MainPID`, `InvocationID`, `NRestarts`, `MemoryMax`, `MemoryPeak`, and
`ControlGroup` with `systemctl --user show laya.service` before and after the
workload. In `/sys/fs/cgroup/<ControlGroup>/`, retain `memory.max`, `memory.peak`,
and `memory.events`. Require the same invocation, no restart/OOM, and a peak below
the 4 GiB limit. Process RSS alone is not this cgroup memory check.

Changing the checkpoint, graph, ORT/QNN binaries, or quantization recipe requires
new correction/cache preparation and fidelity evaluation. To roll back, restore
the backed-up compatible manifest and unit, reload systemd, and restart. Starting
or restarting does not change whether the unit is enabled at login.
