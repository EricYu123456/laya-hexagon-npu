"""Encode folded unit LayerNorm parameters in the HTP-supported static format.

Folded gamma is exactly one and beta is exactly zero. Exported aliases can make
the quantizer encode gamma as a U16 activation, which V68 LayerNorm rejects.
This repair changes only those constant parameters to static U8 gamma / I32
beta; their float32 dequantized values remain exactly one and zero.
"""
import numpy as np
from onnx import TensorProto, helper, numpy_helper


def repair_unit_layernorm_gamma(model):
    """Repair constant-one U16 LayerNorm gamma in place, without recalibration.

    Reject non-unit/dynamic U16 gamma or nonzero beta instead of approximating
    them. Non-U16 LayerNorm parameters and all activation/Conv encodings remain
    unchanged. Remove only newly dead constant-parameter producer chains.
    """
    original_nodes = list(model.graph.node)
    producers = {name: node for node in original_nodes for name in node.output}
    initializers = {item.name: item for item in model.graph.initializer}
    arrays, static_nodes = {}, set()

    def constant(name, visiting=None):
        if name in arrays:
            return arrays[name]
        if name in initializers:
            arrays[name] = numpy_helper.to_array(initializers[name])
            return arrays[name]
        visiting = set() if visiting is None else set(visiting)
        if name in visiting or name not in producers:
            raise ValueError(f"Unit LayerNorm requires a static parameter: {name}")
        visiting.add(name)
        node = producers[name]
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type == "Constant" and isinstance(attrs.get("value"), TensorProto):
            value = numpy_helper.to_array(attrs["value"])
        elif node.op_type == "Identity":
            value = constant(node.input[0], visiting)
        elif node.op_type == "Cast":
            value = constant(node.input[0], visiting).astype(helper.tensor_dtype_to_np_dtype(attrs["to"]))
        elif node.op_type in {"QuantizeLinear", "DequantizeLinear"}:
            value, scale, zero = [constant(item, visiting) for item in node.input]
            if scale.size != 1 or zero.size != 1 or not np.isfinite(scale).all() or not (scale > 0).all():
                raise ValueError(f"Unit LayerNorm only supports finite scalar parameter encodings: {name}")
            if node.op_type == "QuantizeLinear":
                limits = np.iinfo(zero.dtype)
                value = np.clip(np.rint(value.astype(np.float32) / scale) + zero,
                                limits.min, limits.max).astype(zero.dtype)
            else:
                value = (value.astype(np.float32) - zero.astype(np.float32)) * scale.astype(np.float32)
        else:
            raise ValueError(f"Unsupported constant parameter producer {node.op_type}: {name}")
        static_nodes.update(node.output)
        arrays[name] = value
        return value

    plans = {}
    gamma_scale = np.float32(1 / 255)
    if np.float32(255) * gamma_scale != np.float32(1):
        raise ValueError("Static U8 unit gamma must dequantize to exactly float32 one")
    for node in original_nodes:
        if node.op_type != "LayerNormalization":
            continue
        gamma_dq = producers.get(node.input[1])
        if gamma_dq is None or gamma_dq.op_type != "DequantizeLinear":
            continue
        if constant(gamma_dq.input[2]).dtype != np.uint16:
            continue
        gamma = constant(node.input[1])
        if gamma.ndim != 1 or not gamma.size or not np.all(gamma == np.float32(1)):
            raise ValueError(f"{node.name}: U16 gamma must be a constant vector of exact ones")
        if len(node.input) > 2:
            beta = constant(node.input[2])
            if beta.shape != gamma.shape or not np.all(beta == np.float32(0)):
                raise ValueError(f"{node.name}: folded beta must be a constant vector of exact zeros")
        activation_dq = producers.get(node.input[0])
        if activation_dq is None or activation_dq.op_type != "DequantizeLinear":
            raise ValueError(f"{node.name}: expected scalar U16 activation QDQ")
        scale = constant(activation_dq.input[1])
        zero = constant(activation_dq.input[2])
        if (scale.size != 1 or zero.size != 1 or zero.dtype != np.uint16
                or not np.isfinite(scale).all() or not (scale > 0).all()):
            raise ValueError(f"{node.name}: expected finite scalar U16 activation encoding")
        beta_scale = np.float32(scale.reshape(-1)[0]) * gamma_scale
        if not np.isfinite(beta_scale) or beta_scale <= 0:
            raise ValueError(f"{node.name}: invalid static zero-beta scale")
        plans[node.output[0]] = (gamma.shape, beta_scale, gamma_dq.domain)

    occupied = set(initializers) | set(producers) | {node.name for node in original_nodes}
    nodes, records = [], []
    for node in original_nodes:
        if node.output[0] not in plans:
            nodes.append(node)
            continue
        shape, beta_scale, domain = plans[node.output[0]]
        prefix = f"__unit_ln_{len(records)}"
        if any(name.startswith(prefix + "_") for name in occupied):
            raise ValueError(f"Unit LayerNorm repair name collision: {prefix}")
        for label, quantized, scale, zero in (
            ("gamma", np.full(shape, 255, np.uint8), np.array(gamma_scale), np.array(0, np.uint8)),
            ("beta", np.zeros(shape, np.int32), np.array([beta_scale]), np.array(0, np.int32)),
        ):
            names = [f"{prefix}_{label}_{suffix}" for suffix in ("q", "scale", "zero")]
            model.graph.initializer.extend(numpy_helper.from_array(value, name)
                                           for value, name in zip((quantized, scale, zero), names))
            nodes.append(helper.make_node("DequantizeLinear", names, [f"{prefix}_{label}_dq"],
                                          name=f"{prefix}_{label}_DQ", domain=domain))
        node.input[1] = prefix + "_gamma_dq"
        if len(node.input) > 2:
            node.input[2] = prefix + "_beta_dq"
        else:
            node.input.append(prefix + "_beta_dq")
        nodes.append(node)
        records.append({"node": node.name, "channels": int(shape[0]),
                        "gamma_scale": float(gamma_scale), "beta_scale": float(beta_scale)})

    # Do not leave dynamic QDQ aliases for the now-static gamma as unused graph
    # operations. Keep any constant chain still needed by another live consumer.
    removed = 0
    while True:
        used = {item for node in nodes for item in node.input} | {item.name for item in model.graph.output}
        kept = [node for node in nodes if not (set(node.output) <= static_nodes and not set(node.output) & used)]
        if len(kept) == len(nodes):
            break
        removed += len(nodes) - len(kept)
        nodes = kept
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return {"repaired_layer_norms": len(records), "removed_dead_parameter_nodes": removed,
            "gamma": "static U8 code255 scale1/255 zero0; dequantized float32 value exactly1",
            "beta": "static I32 zero; scale equals activation_scale * gamma_scale",
            "nodes": records}
