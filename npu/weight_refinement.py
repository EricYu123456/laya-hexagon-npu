"""Experimental two-term INT8 weights for already-quantized pointwise Convs.

Given the matching FP32 export, replace ``Conv(X, Q8(W))`` with
``Conv(X, Q8(W)) + Conv(X, Q8(W - Q8(W)))``. The two branches use signed
per-channel INT8 weights and U16 activations/results. Graph inputs and final
output quantizers are unchanged. This reduces weight approximation error; it
does not recover activation clipping or guarantee improved accelerator fidelity.

Apply to an uncorrected graph, then rerun the zero-input HTP Conv calibration
for BOTH branches. Corrections from an earlier graph cannot be reused.
"""

import math

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from npu.matmul_refinement import _covering_encoding, _range, _scalar_encoding, _static_shapes


_MARKER = "__weight_refine_"


def _pointwise(node):
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    return (attrs.get("kernel_shape", [1, 1]) == [1, 1]
            and attrs.get("group", 1) == 1
            and attrs.get("strides", [1, 1]) == [1, 1]
            and attrs.get("dilations", [1, 1]) == [1, 1]
            and attrs.get("pads", [0, 0, 0, 0]) == [0, 0, 0, 0]
            and attrs.get("auto_pad", b"NOTSET") in (b"NOTSET", b"VALID"))


def _static_float(name, arrays, producers):
    """Resolve parameter aliases without evaluating arbitrary source operators."""
    seen = set()
    while name not in arrays and name not in seen:
        seen.add(name)
        node = producers.get(name)
        if node is None or node.op_type != "Identity":
            raise ValueError(f"FP32 weight {name!r} is not a static initializer/Identity")
        name = node.input[0]
    if name not in arrays:
        raise ValueError("Cyclic FP32 parameter alias")
    value = arrays[name]
    if value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("Source Conv weights must be finite float32")
    return value


def refine_conv_weights(model, fp32_model, *, node_names=None, activation_l1_bounds=None):
    """Mutate compatible A16U/W8S pointwise Convs and return JSON metadata.

    Match original Conv names to ``fp32_model``; both graphs must be exports of
    the same parameters/recipe. Weight shapes alone cannot verify provenance,
    so the caller must record/verify source hashes. Only scalar U16 activation
    and output encodings and symmetric, axis-0, per-channel INT8 weights qualify.
    Unsupported operators are left alone. A malformed eligible operator or
    missing matching FP32 Conv raises before any mutation. Reapplying is a no-op.

    Low-branch output bounds follow directly from the activation representable
    interval and actual quantized low weights. ``activation_l1_bounds`` can
    provide tighter *proven* bounds on sum(abs(X), channel axis) by Conv name.
    Incorrect bounds can introduce clipping. High-branch output bounds include
    the entire old output interval plus the low-branch bound, so high clipping
    cannot lose cancellation that should survive the original final quantizer.
    """
    selected = None if node_names is None else set(node_names)
    activation_l1_bounds = activation_l1_bounds or {}
    initializers = {item.name: item for item in model.graph.initializer}
    arrays = {name: numpy_helper.to_array(item) for name, item in initializers.items()}
    producers = {name: node for node in model.graph.node for name in node.output}
    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    shapes = _static_shapes(model)
    source_arrays = {item.name: numpy_helper.to_array(item) for item in fp32_model.graph.initializer}
    source_producers = {name: node for node in fp32_model.graph.node for name in node.output}
    source_convs = {}
    for node in fp32_model.graph.node:
        if node.op_type == "Conv":
            if node.name in source_convs:
                raise ValueError(f"Duplicate FP32 Conv name: {node.name}")
            source_convs[node.name] = node
    occupied = set(initializers) | set(producers) | {node.name for node in model.graph.node}
    new_nodes, new_initializers, new_info, records = [], [], [], []
    shared_residuals = {}

    def unique(stem):
        name, suffix = stem, 0
        while name in occupied:
            suffix += 1
            name = f"{stem}_{suffix}"
        occupied.add(name)
        return name

    for node in model.graph.node:
        if (node.op_type != "Conv" or _MARKER in node.name or not _pointwise(node)
                or (selected is not None and node.name not in selected)):
            new_nodes.append(node)
            continue
        adq, wdq = producers.get(node.input[0]), producers.get(node.input[1])
        activation = _scalar_encoding(adq, initializers, np.uint16)
        output_consumers = consumers.get(node.output[0], [])
        oq = output_consumers[0] if len(output_consumers) == 1 else None
        output = _scalar_encoding(oq, initializers, np.uint16)
        if (not activation or not output or adq.op_type != "DequantizeLinear"
                or oq.op_type != "QuantizeLinear" or wdq is None
                or wdq.op_type != "DequantizeLinear" or len(wdq.input) != 3
                or any(name not in arrays for name in wdq.input)):
            new_nodes.append(node)
            continue
        wq, ws, wz = (arrays[name] for name in wdq.input)
        if wq.dtype != np.int8 or wz.dtype != np.int8:
            new_nodes.append(node)
            continue
        axis = next((int(a.i) for a in wdq.attribute if a.name == "axis"), 1)
        if (wq.ndim != 4 or tuple(wq.shape[2:]) != (1, 1) or axis != 0
                or ws.shape != (wq.shape[0],) or wz.shape != ws.shape
                or np.any(wz != 0) or not np.isfinite(ws).all() or np.any(ws <= 0)):
            raise ValueError(f"{node.name}: requires symmetric axis-0 per-channel INT8 weights")
        source = source_convs.get(node.name)
        if source is None or not _pointwise(source):
            raise ValueError(f"No matching pointwise FP32 Conv: {node.name}")
        wf = _static_float(source.input[1], source_arrays, source_producers)
        if wf.shape != wq.shape:
            raise ValueError(f"{node.name}: FP32 and quantized weight shapes differ")
        shape = shapes.get(node.input[0])
        if shape is None or len(shape) != 4 or shape[1] != wq.shape[1]:
            raise ValueError(f"{node.name}: requires static NCHW input shape matching weights")
        output_shape = [shape[0], wq.shape[0], shape[2], shape[3]]
        high_weight = wq.astype(np.float32) * ws[:, None, None, None]
        residual = wf.astype(np.float64) - high_weight.astype(np.float64)
        tolerance = ws[:, None, None, None].astype(np.float64) / 2
        tolerance = tolerance + 16 * np.finfo(np.float32).eps * np.maximum(np.abs(wf), ws[:, None, None, None])
        if np.any(np.abs(residual) > tolerance):
            raise ValueError(f"{node.name}: FP32 weights do not match the coarse INT8 nearest-rounding cells")
        extents = np.max(np.abs(residual), axis=(1, 2, 3))
        if not np.any(extents):
            new_nodes.append(node)
            continue
        # Round scales outward; never clip residual endpoints through float32
        # rounding. Exactly-zero rows use a harmless positive scale.
        low_scale = np.nextafter((extents / 127).astype(np.float32), np.float32(np.inf))
        low_scale[extents == 0] = np.float32(1)
        low_quant = np.clip(np.rint(residual / low_scale[:, None, None, None]), -127, 127).astype(np.int8)
        low_weight = low_quant.astype(np.float32) * low_scale[:, None, None, None]
        flat = low_weight.reshape(wq.shape[0], -1).astype(np.float64)
        positive, negative = np.maximum(flat, 0).sum(1), np.minimum(flat, 0).sum(1)
        amin, amax = _range(activation, 65535)
        low_min = min(0.0, float(np.min(amin * positive + amax * negative)))
        low_max = max(0.0, float(np.max(amax * positive + amin * negative)))
        l1 = activation_l1_bounds.get(node.name)
        if l1 is not None:
            l1 = float(l1)
            if not math.isfinite(l1) or l1 <= 0:
                raise ValueError(f"{node.name}: activation L1 bound must be finite and positive")
            bound = l1 * float(np.abs(flat).max())
            low_min, low_max = max(low_min, -bound), min(low_max, bound)
        # Small arithmetic allowance; not a universal hardware error guarantee.
        margin = 16 * np.finfo(np.float32).eps * max(abs(low_min), abs(low_max), 1e-12)
        low_output_scale, low_output_zero = _covering_encoding(low_min - margin, low_max + margin, 65535)
        low_min_encoded, low_max_encoded = _range((float(low_output_scale), low_output_zero), 65535)
        out_min, out_max = _range(output, 65535)
        high_output_scale, high_output_zero = _covering_encoding(
            out_min - low_max_encoded, out_max - low_min_encoded, 65535)
        prefix = node.name + _MARKER

        def initializer(role, value, dtype):
            name = unique(prefix + role)
            new_initializers.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
            return name

        def qdq(value, role, scale, zero):
            s, z = initializer(role + "_scale", scale, np.float32), initializer(role + "_zero", zero, np.uint16)
            quantized, restored = unique(prefix + role + "_q"), unique(prefix + role + "_dq")
            new_nodes.extend([
                helper.make_node("QuantizeLinear", [value, s, z], [quantized], name=unique(prefix + role + "_Q"), domain=oq.domain),
                helper.make_node("DequantizeLinear", [quantized, s, z], [restored], name=unique(prefix + role + "_DQ"), domain=oq.domain),
            ])
            new_info.extend([helper.make_tensor_value_info(quantized, TensorProto.UINT16, output_shape),
                             helper.make_tensor_value_info(restored, TensorProto.FLOAT, output_shape)])
            return restored

        # CLS/rest branches share projections. Preserve that sharing so neither
        # the ONNX file nor the accelerator compiler stores duplicate matrices.
        residual_key = (tuple(wdq.input), wdq.domain, id(wf))
        low_dq = shared_residuals.get(residual_key)
        if low_dq is None:
            low_q = initializer("weight_q", low_quant, np.int8)
            low_s = initializer("weight_scale", low_scale, np.float32)
            low_z = initializer("weight_zero", np.zeros_like(low_scale), np.int8)
            low_dq = unique(prefix + "weight_dq")
            new_nodes.append(helper.make_node("DequantizeLinear", [low_q, low_s, low_z], [low_dq],
                                              name=unique(prefix + "weight_DQ"), axis=0, domain=wdq.domain))
            new_info.append(helper.make_tensor_value_info(low_dq, TensorProto.FLOAT, list(wq.shape)))
            shared_residuals[residual_key] = low_dq
        high_output, low_output = unique(prefix + "high_output"), unique(prefix + "low_output")
        high_conv = onnx.NodeProto()
        high_conv.CopyFrom(node)
        high_conv.name, high_conv.output[0] = unique(prefix + "high_Conv"), high_output
        new_nodes.append(high_conv)
        high_result = qdq(high_output, "high_output_u16", high_output_scale, high_output_zero)
        low_conv = onnx.NodeProto()
        low_conv.CopyFrom(node)
        low_conv.name, low_conv.output[0] = unique(prefix + "low_Conv"), low_output
        del low_conv.input[:]
        low_conv.input.extend([node.input[0], low_dq])  # Original bias belongs only to the high branch.
        new_nodes.append(low_conv)
        low_result = qdq(low_output, "low_output_u16", low_output_scale, low_output_zero)
        new_nodes.append(helper.make_node("Add", [high_result, low_result], list(node.output), name=unique(prefix + "Add")))
        refined_error = residual - low_weight.astype(np.float64)
        records.append({
            "node": node.name, "high_conv": high_conv.name, "low_conv": low_conv.name,
            "weight_shape": list(wq.shape), "activation_l1_bound": l1,
            "weight_max_error_before": float(np.abs(residual).max()),
            "weight_max_error_after": float(np.abs(refined_error).max()),
            "weight_mean_error_before": float(np.abs(residual).mean()),
            "weight_mean_error_after": float(np.abs(refined_error).mean()),
            "low_output_range": [float(low_min_encoded), float(low_max_encoded)],
            "high_output_range": list(map(float, _range((float(high_output_scale), high_output_zero), 65535))),
        })

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_initializers)
    model.graph.value_info.extend(new_info)
    return {"refined_convs": len(records), "residual_weight_tensors": len(shared_residuals), "nodes": records,
            "requires_fresh_conv_offset_calibration": True,
            "low_output_bound": "Representable activation interval times actual quantized residual weights"}


def refine_files(*, source, fp32, source_manifest, out_dir, fp32_sidecar=None, checkpoint=None):
    """Validate provenance and write a new immutable single-bucket candidate.

    The FP32 export sidecar is required (defaults to the ONNX path with .json).
    Its graph hash, checkpoint identity and input-hash schema are checked. If
    ``checkpoint`` is supplied, its actual bytes must match both recorded hashes.
    The original calibration/configuration files are not read by this transform;
    their recorded hashes are retained as provenance, not independently reverified.
    """
    import json
    import re
    import time
    from pathlib import Path

    from npu.calibrate_conv_offsets import deployment_manifest
    from npu.fidelity_runtime import sha256_file

    started = time.time()
    source, fp32, source_manifest, out_dir = (Path(path).resolve() for path in (source, fp32, source_manifest, out_dir))
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists; choose a new directory: {out_dir}")
    if source.suffix.lower() != ".onnx" or fp32.suffix.lower() != ".onnx":
        raise ValueError("Source and FP32 files must use .onnx extensions")
    sidecar = Path(fp32_sidecar).resolve() if fp32_sidecar else fp32.with_suffix(".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError("FP32 sidecar must contain the export config")

    def digest(value, label):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
            raise ValueError(f"{label} must be a SHA256 digest")
        return value.lower()

    fp32_hash, source_hash = sha256_file(fp32), sha256_file(source)
    if digest(metadata.get("fp32_sha256"), "FP32 sidecar graph hash") != fp32_hash:
        raise ValueError("FP32 sidecar SHA256 does not match --fp32")
    target = out_dir / source.name
    deployment = deployment_manifest(source_manifest, source, target, source_hash)
    manifest = deployment["metadata"]
    bucket = deployment["bucket"]
    checkpoint_hash = digest(manifest.get("checkpoint_sha256"), "Source manifest checkpoint hash")
    if digest(config.get("checkpoint_sha256"), "FP32 checkpoint hash") != checkpoint_hash:
        raise ValueError("FP32 and source manifest checkpoint hashes differ")
    if checkpoint is not None and sha256_file(checkpoint) != checkpoint_hash:
        raise ValueError("Checkpoint file SHA256 does not match the source and FP32 metadata")
    if config.get("length") != int(bucket) or config.get("conv_linear") is not True:
        raise ValueError("FP32 sidecar must describe the same bucket and Conv projection export")
    for field, default in (("mask_penalty", -100), ("zero_pad_embeddings", False)):
        if config.get(field, default) != manifest.get(field, default):
            raise ValueError(f"FP32 and source manifest {field} differ")
    input_hashes = config.get("input_sha256")
    if not isinstance(input_hashes, dict):
        raise ValueError("FP32 sidecar lacks input hashes; regenerate a verified export")
    digest(input_hashes.get("calibration_dataset_sha256"), "Calibration dataset hash")
    model_hashes = input_hashes.get("model_files_sha256")
    required_files = {"encoder/config.json", "rl_agent_config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"}
    if not isinstance(model_hashes, dict) or not required_files.issubset(model_hashes):
        raise ValueError("FP32 sidecar lacks required model/configuration input hashes")
    for name, value in model_hashes.items():
        digest(value, f"Model input hash {name}")

    model, reference = onnx.load(source), onnx.load(fp32)
    # Graph markers prevent a stale or incorrectly relabelled corrected source
    # from silently receiving a second round of backend-specific bias changes.
    if any(item.name.startswith("__htp_offset_") for item in model.graph.initializer):
        raise ValueError("Source graph already contains HTP Conv offset corrections")
    if any(_MARKER in node.name for node in model.graph.node):
        raise ValueError("Source graph already contains weight refinement")
    result = refine_conv_weights(model, reference)
    if not result["refined_convs"]:
        raise ValueError("No compatible Conv weights were refined")
    onnx.checker.check_model(model)
    # A dedicated new directory makes partial artifacts after interruption
    # visible and prevents later retries from overwriting prior evidence.
    out_dir.mkdir(parents=True, exist_ok=False)
    onnx.save(model, target)
    target_hash = sha256_file(target)
    result.update({
        "source_qdq": str(source), "source_qdq_sha256": source_hash,
        "source_fp32": str(fp32), "source_fp32_sha256": fp32_hash,
        "source_manifest_sha256": deployment["source_manifest_sha256"],
        "fp32_sidecar_sha256": sha256_file(sidecar), "fp32_export_config": config,
        "checkpoint_sha256": checkpoint_hash, "checkpoint_bytes_verified": checkpoint is not None,
        "source_input_hashes_verified_against_files": False,
        "target_sha256": target_hash, "seconds": time.time() - started,
    })
    manifest["buckets"], manifest["model_sha256"] = {bucket: target.name}, {bucket: target_hash}
    manifest["weight_refinement"] = "two_term_int8"
    for path, content in ((out_dir / "manifest.json", manifest), (out_dir / "weight-refinement.json", result)):
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(content, indent=2, allow_nan=False) + "\n")
    return result


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Uncorrected A16U/W8S QDQ graph")
    parser.add_argument("--fp32", required=True, help="Matching verified FP32 export")
    parser.add_argument("--source-manifest", required=True, help="Manifest binding source graph and checkpoint")
    parser.add_argument("--out-dir", required=True, help="New, nonexistent output directory")
    parser.add_argument("--fp32-sidecar", help="Export sidecar; default: FP32 path with .json extension")
    parser.add_argument("--checkpoint", help="Also verify the original checkpoint file bytes")
    result = refine_files(**vars(parser.parse_args()))
    print(json.dumps({key: value for key, value in result.items() if key not in {"nodes", "fp32_export_config"}}, indent=2))


if __name__ == "__main__":
    main()
