"""Keep LayerNorm gamma in U16 elementwise arithmetic instead of U8 LN weights.

The HTP LayerNorm receives exactly unit U8 gamma and zero I32 beta. Its output
is U16 quantized, then multiplied/added to separately U16-quantized original
FP32 affine parameters. This is a precision improvement, not exact arithmetic.
It needs hardware qualification; CPU QDQ accuracy does not certify HTP.
"""
from __future__ import annotations

import math

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from npu.matmul_refinement import _covering_encoding, _scalar_encoding, _static_shapes


def _constant_reader(model):
    initializers = {item.name: item for item in model.graph.initializer}
    producers = {name: node for node in model.graph.node for name in node.output}
    cache, visited = {}, set()

    def read(name, stack=()):
        if name in cache:
            return cache[name]
        if name in initializers:
            value = numpy_helper.to_array(initializers[name])
        else:
            if name in stack or name not in producers:
                raise ValueError(f"LayerNorm affine parameter is not static: {name}")
            node = producers[name]
            attrs = {item.name: helper.get_attribute_value(item) for item in node.attribute}
            if node.op_type == "Constant" and isinstance(attrs.get("value"), TensorProto):
                value = numpy_helper.to_array(attrs["value"])
            elif node.op_type == "Identity":
                value = read(node.input[0], (*stack, name))
            elif node.op_type == "Cast":
                value = read(node.input[0], (*stack, name)).astype(helper.tensor_dtype_to_np_dtype(attrs["to"]))
            elif node.op_type in {"QuantizeLinear", "DequantizeLinear"}:
                value, scale, zero = [read(item, (*stack, name)) for item in node.input]
                if scale.size != 1 or zero.size != 1 or not np.isfinite(scale).all() or not (scale > 0).all():
                    raise ValueError(f"Only scalar parameter encodings are supported: {name}")
                if node.op_type == "QuantizeLinear":
                    limits = np.iinfo(zero.dtype)
                    value = np.clip(np.rint(value.astype(np.float32) / scale) + zero,
                                    limits.min, limits.max).astype(zero.dtype)
                else:
                    value = (value.astype(np.float32) - zero.astype(np.float32)) * scale.astype(np.float32)
            else:
                raise ValueError(f"Unsupported static parameter producer: {node.op_type} ({name})")
            visited.update(node.output)
        cache[name] = value
        return value

    return read, visited


def refine_layernorm_affines(model, reference_model, *, node_names=None):
    """Rewrite selected learned-affine LayerNorms, using matching FP32 node names.

    All supported learned LayerNorms are selected by default. Existing unit
    affine norms and previously rewritten norms are skipped. Inputs must be
    scalar U16 QDQ. The last feature axis and vector affine parameters are
    required. Original input/output encodings and graph IO stay unchanged.

    The extra unit-output QDQ covers +-sqrt(channels-1), the mathematical bound
    for unit LayerNorm (positive epsilon), with a small arithmetic margin. No
    calibration or evaluation examples determine these new encodings. Reference
    gamma/beta are quantized directly to U16; they are never reconstructed from
    the original U8 parameter encoding. Validation precedes graph mutation.
    """
    selected = None if node_names is None else set(node_names)
    def named_norms(graph):
        nodes = [node for node in graph.node if node.op_type == "LayerNormalization"]
        names = [node.name for node in nodes]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("LayerNorm nodes require unique nonempty names")
        return {node.name: node for node in nodes}

    reference_nodes = named_norms(reference_model.graph)
    named_norms(model.graph)
    original_nodes = list(model.graph.node)
    initializers = {item.name: item for item in model.graph.initializer}
    producers = {name: node for node in original_nodes for name in node.output}
    read_reference, _ = _constant_reader(reference_model)
    read_current, old_parameter_outputs = _constant_reader(model)
    shapes = _static_shapes(model)
    occupied = set(initializers) | set(producers) | {node.name for node in original_nodes}
    plans, skipped = {}, []
    for node in original_nodes:
        if node.op_type != "LayerNormalization" or (selected is not None and node.name not in selected):
            continue
        if "__affine16_" in node.output[0]:
            skipped.append({"node": node.name, "reason": "already_refined"})
            continue
        reference = reference_nodes.get(node.name)
        if reference is None:
            raise ValueError(f"No matching FP32 LayerNorm: {node.name}")
        attrs = {item.name: helper.get_attribute_value(item) for item in node.attribute}
        reference_attrs = {item.name: helper.get_attribute_value(item) for item in reference.attribute}
        if attrs != reference_attrs or int(attrs.get("axis", -1)) != -1:
            raise ValueError(f"{node.name}: matching last-axis LayerNorm attributes required")
        epsilon = float(attrs.get("epsilon", 1e-5))
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError(f"{node.name}: positive finite epsilon required")
        gamma = np.asarray(read_reference(reference.input[1]), dtype=np.float32)
        beta = (np.asarray(read_reference(reference.input[2]), dtype=np.float32)
                if len(reference.input) > 2 else np.zeros_like(gamma))
        if (gamma.ndim != 1 or gamma.size < 2 or beta.shape != gamma.shape
                or not np.isfinite(gamma).all() or not np.isfinite(beta).all()):
            raise ValueError(f"{node.name}: finite gamma/beta vectors with at least two channels required")
        for value_name in (node.input[0], node.output[0]):
            shape = shapes.get(value_name)
            if shape and shape[-1] != gamma.size:
                raise ValueError(f"{node.name}: affine vector does not match the static feature dimension")
        if np.all(gamma == 1) and np.all(beta == 0):
            skipped.append({"node": node.name, "reason": "unit_affine"})
            continue
        input_dq = producers.get(node.input[0])
        encoding = _scalar_encoding(input_dq, initializers, np.uint16)
        if encoding is None or input_dq.op_type != "DequantizeLinear":
            raise ValueError(f"{node.name}: scalar U16 QDQ input required")
        # Evaluate current parameter chains only to prune newly dead aliases.
        current_gamma = read_current(node.input[1])
        if current_gamma.shape != gamma.shape:
            raise ValueError(f"{node.name}: reference/current gamma shape mismatch")
        if len(node.input) > 2 and read_current(node.input[2]).shape != beta.shape:
            raise ValueError(f"{node.name}: reference/current beta shape mismatch")
        prefix = node.name + "__affine16_"
        if any(name.startswith(prefix) for name in occupied):
            raise ValueError(f"{node.name}: affine refinement naming collision")
        plans[node.name] = (gamma, beta, encoding[0], input_dq.domain, prefix)

    new_nodes, new_initializers, new_info, records = [], [], [], []
    unit_gamma_scale = np.float32(1 / 255)
    for node in original_nodes:
        if node.name not in plans:
            new_nodes.append(node)
            continue
        gamma, beta, input_scale, domain, prefix = plans[node.name]
        output_name = node.output[0]
        output_shape = shapes.get(output_name) or shapes.get(node.input[0])

        def static_dq(role, quantized, scale, zero):
            names = [prefix + role + suffix for suffix in ("_q", "_scale", "_zero")]
            for name, value in zip(names, (quantized, scale, zero)):
                new_initializers.append(numpy_helper.from_array(np.asarray(value), name))
            restored = prefix + role + "_dq"
            new_nodes.append(helper.make_node("DequantizeLinear", names, [restored], name=prefix + role + "_DQ", domain=domain))
            return restored

        def affine_dq(role, values):
            scale, zero = _covering_encoding(float(values.min()), float(values.max()), 65535)
            codes = np.clip(np.rint(values / scale) + zero, 0, 65535).astype(np.uint16)
            restored = (codes.astype(np.float32) - np.float32(zero)) * scale
            name = static_dq(role, codes, np.asarray(scale), np.asarray(zero, np.uint16))
            return name, restored, float(np.max(np.abs(restored - values)))

        def dynamic_qdq(role, value, bound):
            scale, zero = _covering_encoding(-bound, bound, 65535)
            scale_name, zero_name = prefix + role + "_scale", prefix + role + "_zero"
            new_initializers.extend([numpy_helper.from_array(np.asarray(scale), scale_name),
                                     numpy_helper.from_array(np.asarray(zero, np.uint16), zero_name)])
            quantized, restored = prefix + role + "_q", prefix + role + "_dq"
            new_nodes.extend([
                helper.make_node("QuantizeLinear", [value, scale_name, zero_name], [quantized], name=prefix + role + "_Q", domain=domain),
                helper.make_node("DequantizeLinear", [quantized, scale_name, zero_name], [restored], name=prefix + role + "_DQ", domain=domain),
            ])
            if output_shape:
                new_info.extend([helper.make_tensor_value_info(quantized, TensorProto.UINT16, output_shape),
                                 helper.make_tensor_value_info(restored, TensorProto.FLOAT, output_shape)])
            return restored, float(scale)

        unit_gamma = static_dq("unit_gamma", np.full(gamma.shape, 255, np.uint8),
                               np.asarray(unit_gamma_scale), np.asarray(0, np.uint8))
        unit_beta = static_dq("unit_beta", np.zeros(beta.shape, np.int32),
                              np.asarray([np.float32(input_scale) * unit_gamma_scale]), np.asarray(0, np.int32))
        precise_gamma, gamma_restored, gamma_error = affine_dq("gamma", gamma)
        precise_beta, _, beta_error = affine_dq("beta", beta) if np.any(beta != 0) else (None, None, 0.0)
        unit_node = type(node)()
        unit_node.CopyFrom(node)
        unit_node.input[1] = unit_gamma
        if len(unit_node.input) > 2:
            unit_node.input[2] = unit_beta
        else:
            unit_node.input.append(unit_beta)
        unit_node.output[0] = prefix + "unit_raw"
        new_nodes.append(unit_node)
        unit_bound = math.sqrt(gamma.size - 1) * (1 + 1e-5)
        unit, unit_scale = dynamic_qdq("unit", unit_node.output[0], unit_bound)
        product = output_name if precise_beta is None else prefix + "product_raw"
        new_nodes.append(helper.make_node("Mul", [unit, precise_gamma], [product], name=prefix + "Mul"))
        if precise_beta is not None:
            product, _ = dynamic_qdq("product", product, unit_bound * float(np.abs(gamma_restored).max()))
            new_nodes.append(helper.make_node("Add", [product, precise_beta], [output_name], name=prefix + "Add"))
        records.append({"node": node.name, "channels": int(gamma.size), "unit_output_bound": unit_bound,
                        "unit_output_scale": unit_scale, "gamma_max_abs_error": gamma_error,
                        "beta_max_abs_error": beta_error, "nonzero_beta": precise_beta is not None})

    # Remove only old, static parameter chains that have lost their consumers.
    removed = 0
    while True:
        used = {name for node in new_nodes for name in node.input} | {item.name for item in model.graph.output}
        kept = [node for node in new_nodes if not (set(node.output) <= old_parameter_outputs and not set(node.output) & used)]
        if len(kept) == len(new_nodes):
            break
        removed += len(new_nodes) - len(kept)
        new_nodes = kept
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_initializers)
    model.graph.value_info.extend(new_info)
    return {"refined_layer_norms": len(records), "nodes": records, "skipped": skipped,
            "removed_dead_parameter_nodes": removed, "parameter_source": "matching FP32 graph",
            "unit_range_rule": "sqrt(channels-1) * (1+1e-5)", "affine_precision": "U16 per-tensor"}


def refine_files(*, source, fp32, source_manifest, out_dir, fp32_sidecar=None,
                 checkpoint=None, weight_provenance=None, build_metadata=None,
                 build_manifest=None, expected_layernorms=None):
    """Write a new candidate after checking the existing export/provenance chain.

    A weight-refined source requires its weight-refinement.json (by default next
    to its manifest). Build metadata and the original build manifest are retained
    when available beside the FP32 export, or may be explicitly supplied. This
    does not reverify the original calibration/tokenizer files; their existing
    hashes and complete metadata are retained. No existing output is overwritten.
    """
    import json
    from pathlib import Path
    import re
    import shutil
    import time

    import onnx
    from npu.calibrate_conv_offsets import deployment_manifest
    from npu.fidelity_runtime import sha256_file

    started = time.time()
    source, fp32, source_manifest, out_dir = (Path(path).resolve() for path in (source, fp32, source_manifest, out_dir))
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists; choose a new directory: {out_dir}")
    if source.suffix.lower() != ".onnx" or fp32.suffix.lower() != ".onnx":
        raise ValueError("Source and FP32 files must use .onnx extensions")
    if expected_layernorms is not None and (type(expected_layernorms) is not int or expected_layernorms <= 0):
        raise ValueError("expected_layernorms must be a positive integer")

    def digest(value, label):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
            raise ValueError(f"{label} must be a SHA256 digest")
        return value.lower()

    source_hash, fp32_hash = sha256_file(source), sha256_file(fp32)
    target = out_dir / source.name
    deployment = deployment_manifest(source_manifest, source, target, source_hash)
    manifest, bucket = deployment["metadata"], deployment["bucket"]
    if "layernorm_affine_refinement" in manifest:
        raise ValueError("Source manifest already declares LayerNorm affine refinement")
    sidecar_path = Path(fp32_sidecar).resolve() if fp32_sidecar else fp32.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    config = sidecar.get("config")
    if not isinstance(config, dict):
        raise ValueError("FP32 sidecar must contain the export config")
    if digest(sidecar.get("fp32_sha256"), "FP32 sidecar graph hash") != fp32_hash:
        raise ValueError("FP32 sidecar SHA256 does not match --fp32")
    checkpoint_hash = digest(manifest.get("checkpoint_sha256"), "Source checkpoint hash")
    if digest(config.get("checkpoint_sha256"), "FP32 checkpoint hash") != checkpoint_hash:
        raise ValueError("FP32 and source manifest checkpoint hashes differ")
    if checkpoint is not None and sha256_file(checkpoint) != checkpoint_hash:
        raise ValueError("Checkpoint file SHA256 does not match the source and FP32 metadata")
    if config.get("length") != int(bucket):
        raise ValueError("FP32 sidecar must describe the same bucket")
    for field, default in (("mask_penalty", -100), ("zero_pad_embeddings", False)):
        if config.get(field, default) != manifest.get(field, default):
            raise ValueError(f"FP32 and source manifest {field} differ")
    input_hashes = config.get("input_sha256")
    if not isinstance(input_hashes, dict):
        raise ValueError("FP32 sidecar lacks input hashes; regenerate a verified export")
    digest(input_hashes.get("calibration_dataset_sha256"), "Calibration dataset hash")
    model_hashes = input_hashes.get("model_files_sha256")
    required = {"encoder/config.json", "rl_agent_config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"}
    if not isinstance(model_hashes, dict) or not required.issubset(model_hashes):
        raise ValueError("FP32 sidecar lacks required model/configuration input hashes")
    for name, value in model_hashes.items():
        digest(value, f"Model input hash {name}")

    artifacts = {"fp32-export.json": sidecar_path, "source-weight-manifest.json": source_manifest}
    weight_record = None
    if manifest.get("weight_refinement") is not None or weight_provenance is not None:
        weight_path = Path(weight_provenance).resolve() if weight_provenance else source_manifest.parent / "weight-refinement.json"
        weight_record = json.loads(weight_path.read_text(encoding="utf-8"))
        for field, expected in (("target_sha256", source_hash), ("source_fp32_sha256", fp32_hash),
                                ("fp32_sidecar_sha256", sha256_file(sidecar_path)), ("checkpoint_sha256", checkpoint_hash)):
            if digest(weight_record.get(field), f"Weight provenance {field}") != expected:
                raise ValueError(f"Weight provenance {field} does not match the requested source")
        if weight_record.get("fp32_export_config") != config:
            raise ValueError("Weight and affine refinements require identical FP32 export configuration")
        artifacts["weight-refinement.json"] = weight_path

    for name, explicit, default in (
        ("build-metadata.json", build_metadata, fp32.parent / f"backbone_{bucket}_a16w8.json"),
        ("source-build-manifest.json", build_manifest, fp32.parent / "manifest.json"),
    ):
        path = Path(explicit).resolve() if explicit is not None else default
        if explicit is not None or path.is_file():
            if not path.is_file():
                raise FileNotFoundError(path)
            record = json.loads(path.read_text(encoding="utf-8"))
            if weight_record is not None:
                if name == "source-build-manifest.json" and sha256_file(path) != weight_record.get("source_manifest_sha256"):
                    raise ValueError("Build manifest changed after weight refinement")
                if name == "build-metadata.json" and record.get("model_sha256") != weight_record.get("source_qdq_sha256"):
                    raise ValueError("Build metadata does not describe the weight refinement's source graph")
            artifacts[name] = path

    model, reference = onnx.load(source), onnx.load(fp32)
    if any(item.name.startswith("__htp_offset_") for item in model.graph.initializer):
        raise ValueError("Source graph already contains HTP Conv offset corrections")
    if any("__affine16_" in name for node in model.graph.node for name in node.output):
        raise ValueError("Source graph already contains LayerNorm affine refinement")
    result = refine_layernorm_affines(model, reference)
    count = result["refined_layer_norms"]
    if not count or (expected_layernorms is not None and count != expected_layernorms):
        raise ValueError(f"Unexpected refined LayerNorm count: {count}; expected {expected_layernorms or 'at least one'}")
    onnx.checker.check_model(model)
    out_dir.mkdir(parents=True, exist_ok=False)
    onnx.save(model, target)
    for name, path in artifacts.items():
        shutil.copyfile(path, out_dir / name)
    target_hash = sha256_file(target)
    result.update({
        "source_qdq": str(source), "source_qdq_sha256": source_hash,
        "source_fp32": str(fp32), "source_fp32_sha256": fp32_hash,
        "source_manifest_sha256": deployment["source_manifest_sha256"],
        "checkpoint_sha256": checkpoint_hash, "checkpoint_bytes_verified": checkpoint is not None,
        "source_input_hashes_verified_against_files": False, "target_sha256": target_hash,
        "retained_metadata_sha256": {name: sha256_file(out_dir / name) for name in artifacts},
        "seconds": time.time() - started,
    })
    manifest["buckets"], manifest["model_sha256"] = {bucket: target.name}, {bucket: target_hash}
    manifest["layernorm_affine_refinement"] = "unit_ln_u16_affine"
    for path, content in ((out_dir / "manifest.json", manifest), (out_dir / "affine-refinement.json", result)):
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(content, indent=2, allow_nan=False) + "\n")
    return result


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Uncorrected QDQ graph, optionally weight-refined")
    parser.add_argument("--fp32", required=True, help="Matching verified FP32 export")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--out-dir", required=True, help="New, nonexistent output directory")
    parser.add_argument("--fp32-sidecar", help="Default: FP32 path with .json extension")
    parser.add_argument("--checkpoint", help="Also verify the original checkpoint bytes")
    parser.add_argument("--weight-provenance", help="Default for weight-refined input: beside its manifest")
    parser.add_argument("--build-metadata", help="Optional original quantization metadata; retained when found beside FP32")
    parser.add_argument("--build-manifest", help="Optional original build manifest; retained when found beside FP32")
    parser.add_argument("--expected-layernorms", type=int)
    result = refine_files(**vars(parser.parse_args()))
    print(json.dumps({key: value for key, value in result.items() if key not in {"nodes", "skipped"}}, indent=2))


if __name__ == "__main__":
    main()
