# HTP Conv offset calibration

`calibrate_conv_offsets.py` measures the difference between CPU QDQ and QNN HTP
outputs when every Conv input is zero. It adds the negative measured difference
to each Conv's existing bias. This corrects a repeatable backend offset without
fitting evaluation examples.

The corrected model is specific to the measured ORT/QNN binaries and HTP
architecture. Its CPU behavior intentionally changes. Compare the corrected
model **on HTP** with the original Laya reference; CPU inference on the corrected
graph is not an accuracy substitute.

## Workflow

Run these commands from the repository root. `prepare` and `apply` need the
ONNX build dependencies and work on a host or WSL. `run` requires the target Pi,
the matching QNN environment, and exclusive use of its accelerator.

1. Prepare probes from an **uncorrected** static QDQ model into a fresh output
   directory:

   ```bash
   python -m npu.calibrate_conv_offsets prepare \
     --source .work/refined32/backbone_768_a16w8.onnx \
     --out .work/conv_offsets_768 \
     --validation-count 3
   ```

2. On the Pi, run zero-input CPU and strict HTP probes:

   ```bash
   source npu/env.sh
   .venv/bin/python -m npu.calibrate_conv_offsets run \
     --out .work/conv_offsets_768
   ```

3. Write a new corrected artifact:

   ```bash
   python -m npu.calibrate_conv_offsets apply \
     --source .work/refined32/backbone_768_a16w8.onnx \
     --out .work/conv_offsets_768 \
     --target .work/corrected_768/backbone_768_a16w8.onnx \
     --source-manifest .work/refined32/manifest.json
   ```

`apply` refuses an existing target or a path that would overwrite the source.
The source argument may use a different host path, but its SHA256 must match the
exact file used for preparation. Each sequence-length bucket needs its own
preparation, measurements, corrected model, and provenance. In particular, do
not apply a 768-token model's calibration to a 1024-token graph.

`--source-manifest` is optional. When supplied, each source bucket path is
resolved relative to that manifest, exactly one path must match `--source`, and
its declared SHA256 must match the source model. Already-corrected source
manifests and an existing destination `manifest.json` are rejected. The emitted
manifest contains only the selected corrected bucket while preserving checkpoint,
padding, RHS refinement, mask, and other policy fields. Prepare/correct each
bucket independently before assembling a multi-bucket deployment manifest.

## Artifacts and deployment

The calibration directory contains:

| File | Contents |
| --- | --- |
| `zero_probes.onnx` | Independent Convs with original weights and encodings |
| `probe_manifest.json` | Source/probe hashes, Conv metadata, sampled shape controls |
| `zero_outputs.npz` | Measured CPU and HTP zero outputs |
| `correction_biases.npz` | Per-channel `CPU_zero - HTP_zero` corrections |
| `calibration_results.json` | Offsets, shape checks, binary fingerprints, array hash |

The corrected ONNX file has an adjacent `.offsets.json` provenance file. With
`--source-manifest`, the tool also emits a ready-to-use `manifest.json` beside
the corrected model. It updates the bucket path and model SHA256, and embeds
the provenance under `htp_conv_offset_correction["768"]`. Without this option,
make those manifest updates before deployment. Run the normal fidelity benchmark
on the corrected artifact. The runtime checks the
corrected model hash and its recorded ORT version, QNN package version, HTP
architecture, backend, host stub, and DSP skeleton hashes before using source
graphs or context caches. Deploy the model and matching provenance together.

## Checks and limitations

Supported Convs are static NCHW batch-one, pointwise 1×1, group-one, stride-one
Convs with unsigned 16-bit activation/output and signed 8-bit weights. Existing
static biases are retained and adjusted. New biases use signed INT32 with
per-channel scale `activation_scale * weight_scale`. Overflow, non-finite
values, wrong source hashes, altered calibration arrays, and incomplete shape
checks cause an error; biases are never silently clipped.

The compact probe uses spatial size one for every Conv. By default, the first
three Convs with a larger original spatial shape are also measured at that
shape. Application requires their offset vectors to agree within one-quarter
of an output quantization step plus floating-point tolerance. This is a sampled
kernel-shape check, not proof for every Conv. Increase `--validation-count` when
changing a graph layout or backend.

Offsets are measured after the original output quantizer, so each correction
has up to one output-step uncertainty. A zero output at a quantizer's saturation
boundary can hide part of an offset. Full-model HTP validation remains required
even when all zero-input checks pass.

New calibration records fingerprint the host stub and DSP skeleton in addition
to the backend. Version-one records from the initial experiment remain usable:
their exact source/probe hashes, Conv records, corrections, and spatial checks
are still verified. Missing historical auxiliary hashes are not invented or
filled with values from a later installation. Recalibrate to obtain the fuller
fingerprint after changing QNN binaries.

Run the CPU-only contract tests with:

```bash
python -m unittest tests.test_conv_offset_calibration -v
```
