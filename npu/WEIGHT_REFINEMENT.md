# Two-term INT8 Conv weight refinement

`python -m npu.weight_refinement` builds a candidate with finer
static projection weights while retaining the HTP-compatible A16U/W8S Conv
operators. It requires an **uncorrected** QDQ graph and its matching verified
FP32 export. It does not need training data, evaluation labels or the NPU during
the transform. It is not a claim that the candidate meets a fidelity threshold.

The complete 2026-10-03 recipe combines this transform with feature-band/CLS
separation, compatible LayerNorm folding, U16 affine refinement and fresh HTP
Conv-offset correction. That fixed recipe passed all nine historical NPU suites
and four declared subsets; see the [qualification audit](../reports/qualification-2026-10-03/one-percent-audit.json)
and [reproduction commands](../docs/build-fidelity.md). The maximum suite/subset
decision mismatch rate was 0.6% and maximum mean TV was 0.622212%. Weight
refinement alone did not lower mean TV in the [fixed development comparison](../reports/precision-development-2026-10-03/).
New source graphs or parameter choices still require their own qualification.

For each supported pointwise Conv, the transform computes

```text
W_high = dequantize(original INT8 weights)
W_low = per-output-channel INT8 quantization of (W_FP32 - W_high)
Y = Q16(Conv(X, W_high, original_bias)) + Q16(Conv(X, W_low))
```

The original final output quantizer consumes this sum. Inputs, masks, graph
outputs and original static weights are preserved. The bias belongs only to
the high branch. CLS/rest projections retain shared residual weight storage.
The transform supports static NCHW, unit-stride, ungrouped 1×1 Conv with scalar
U16 activation/output encodings and symmetric axis-0 per-channel INT8 weights.

The residual result range is bounded using the activation's entire
representable interval and the actual quantized residual weights. The high
result range expands to retain cancellation before the original output
quantizer. This avoids a failure where a high term saturates before a low term
can bring the sum back into range. Both new results remain U16; the additional
rounding can still limit accuracy, especially with wide activation ranges.
The Python API optionally accepts proven activation L1 bounds. The CLI uses
the conservative representable-interval bound and does not estimate bounds
from evaluation examples. The qualified recipe uses this CLI default. Tighter
LayerNorm-derived bounds tested in a separate CPU-only experiment were not
selected; operator-formula bounds do not by themselves certify approximate
HTP kernels or downstream Conv accumulation.

## Build

Use the x86/WSL build environment from
[`requirements-fidelity-build.txt`](../requirements-fidelity-build.txt). The
transform itself requires NumPy and ONNX; CPU contract tests also need ONNX
Runtime. No ARM cross compiler is needed.

```bash
python -m npu.weight_refinement \
  --source .work/features32_folded768/backbone_768_a16w8.onnx \
  --fp32 .work/features32_folded768/backbone_768_fp32.onnx \
  --source-manifest .work/features32_folded768/manifest.json \
  --checkpoint models/multilingual/model.safetensors \
  --out-dir .work/features32_weight16_768
```

The output directory must not exist. It receives the new ONNX graph,
`manifest.json`, and `weight-refinement.json`. Source files are never overwritten.
An interrupted build can leave a partial directory; retry with a new directory.

The required FP32 export sidecar defaults to `backbone_768_fp32.json`; use
`--fp32-sidecar` to supply another path. Preflight verifies:

- The source graph's checksum and resolved path match its manifest.
- The FP32 graph's checksum matches its sidecar.
- The two artifacts declare the same checkpoint, bucket, mask penalty and
  padding contract, and the FP32 export uses Conv projections.
- The sidecar contains the calibration and model/configuration input hashes.
- Every FP32 weight lies within the original INT8 nearest-rounding cell, with
  a float32 arithmetic allowance. Matching names and shapes alone are
  insufficient. Saturated weights outside those cells are rejected.
- Neither the source manifest nor graph carries an existing HTP Conv offset
  correction; an already refined graph is also rejected by the CLI.

`--checkpoint` additionally verifies the actual checkpoint bytes. Without it,
the two recorded checkpoint hashes must still agree, but the checkpoint file
is not independently read. The transform retains the FP32 sidecar's input hashes;
it does not reread the original dataset, tokenizer or configuration files.
Provenance distinguishes these cases explicitly. A coarse quantization-cell
check helps detect incompatible recipes but is not a replacement for trusted
matching-export provenance.

For the 1024 bucket, supply its own folded FP32 export, uncorrected QDQ graph,
and manifest. Do not mix buckets or folding recipes.

For the qualified combined recipe, run the affine refinement step in the
[build guide](../docs/build-fidelity.md) before preparing probes. Its final
uncorrected source is `.work/features32_precision16_768`, or the corresponding
1024 directory. The standalone weight-only calibration example below illustrates
the same calibration contract but is not the complete qualified recipe.

## Hardware calibration and qualification

Every refined projection becomes two Convs. **Both branches require fresh HTP
zero-input offset calibration.** Corrections and compiled contexts from the old
graph cannot be reused directly.

```bash
python -m npu.calibrate_conv_offsets prepare \
  --source .work/features32_weight16_768/backbone_768_a16w8.onnx \
  --out .work/features32_weight16_768/offset-probes
```

Transfer the candidate and probes to the Pi, then follow
[`CONV_OFFSET_CALIBRATION.md`](CONV_OFFSET_CALIBRATION.md). Source `npu/env.sh`
before all QNN commands. Apply the measured correction to a separate graph,
with `--source-manifest` pointing at the refinement candidate's manifest.
Only then evaluate the full graph with strict HTP execution and unchanged
original Laya inputs. CPU QDQ improvement and isolated probe success do not
certify full-model hardware accuracy or the one-percent goal.

An affine-only follow-up can reuse the original measured calibration if its
**entire prepared probe file is byte-identical**, its full Conv records and
spatial-validation IDs are identical, and the current Pi runtime matches the
measured backend. This does not permit copying correction files by hand or
reusing another bucket. Both source graphs must remain available. After preparing
the recipient probes, run on the Pi:

```bash
source npu/env.sh
python -m npu.reuse_conv_calibration \
  --from .work/features32_weight16_768/offset-probes \
  --out .work/features32_precision16_768/offset-probes \
  --from-source .work/features32_weight16_768/backbone_768_a16w8.onnx \
  --source .work/features32_precision16_768/backbone_768_a16w8.onnx
```

The reuse command revalidates the donor's raw CPU/HTP arrays, corrections and
spatial checks, verifies source/probe/result/correction hashes, and records the
original measurement hashes in the recipient result and corrected-model
provenance. The recipient must have no existing measurement files. Reuse chains
are rejected; always point to the original directly measured donor. This saves
recompiling an identical zero-probe graph; full-model hardware qualification is
still required after every change.

This adds Conv work and approximately doubles unique projection-weight storage.
Compilation memory, latency, activation quantization and HTP arithmetic must be
measured. Weight decomposition does not repair LayerNorm parameter precision,
activation clipping, calibration-range errors or unrelated operator errors.
The combined qualified recipe's 768-token development latency was 84.3% higher
than the September 29 baseline. Measured compilation took 21 min 25.39 s for
768 and 1 h 22 min 44 s for 1024, with peak process RSS of 6.5412 and 6.5901 GiB;
these are offline build measurements, not service memory measurements. See the
[development](../reports/precision-development-2026-10-03/) and
[build](../reports/precision-build-2026-10-03/) evidence bundles.

Run CPU contracts with:

```bash
python -m unittest discover -s tests -p test_weight_refinement.py -v
```

The tests cover numerical improvement, cancellation at both clipping limits,
single application of bias, shared matrices, source compatibility, immutable
outputs, checksum/checkpoint rejection and compatibility with Conv probes.
