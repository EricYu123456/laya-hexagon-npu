"""Experimental post-QDQ refinement of dynamic U16-to-U8 MatMul RHS inputs.

By default the original U8 term is retained and a quantized residual is added:
    A @ high + A @ Q8(rhs16 - high)
This reduces RHS quantization error; it is not exact after quantization. Both
MatMuls remain U16 x U8, and their partial results and the Sub use U16 outputs.
No hardware accuracy or V68 compilation guarantee is implied by this transform.
"""

import math

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _scalar_encoding(node, initializers, dtype):
    if node is None or node.op_type not in {"QuantizeLinear", "DequantizeLinear"} or len(node.input) != 3:
        return None
    if any(attr.name in {"axis", "block_size"} for attr in node.attribute):
        return None
    if node.input[1] not in initializers or node.input[2] not in initializers:
        return None
    scale = numpy_helper.to_array(initializers[node.input[1]])
    zero = numpy_helper.to_array(initializers[node.input[2]])
    if scale.size != 1 or zero.size != 1 or zero.dtype != np.dtype(dtype):
        return None
    scale, zero = float(scale.reshape(-1)[0]), int(zero.reshape(-1)[0])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid quantization scale in {node.name}")
    return scale, zero


def _range(encoding, maximum):
    scale, zero = encoding
    return -zero * scale, (maximum - zero) * scale


def _covering_encoding(low, high, maximum):
    """Include zero and both bounds, also after integer zero-point rounding."""
    low, high = min(float(low), 0.0), max(float(high), 0.0)
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("Non-finite residual range")
    if low == high:
        return np.float32(1.0), 0
    scale = (high - low) / maximum
    zero = int(np.clip(np.rint(-low / scale), 0, maximum))
    if low < 0 < high:
        zero = min(max(zero, 1), maximum - 1)
    scale = max((-low / zero if zero else 0.0), (high / (maximum - zero) if zero < maximum else 0.0))
    # Round the float32 scale outward so declared clipping bounds stay valid.
    scale = np.nextafter(np.float32(scale), np.float32(np.inf))
    return scale, zero


def _static_shapes(model):
    shapes = {}
    for item in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        dims = item.type.tensor_type.shape.dim
        if dims and all(dim.HasField("dim_value") and dim.dim_value > 0 for dim in dims):
            shapes[item.name] = [dim.dim_value for dim in dims]
    for item in model.graph.initializer:
        if item.dims:
            shapes[item.name] = list(item.dims)
    for node in model.graph.node:
        # Microsoft's U16 QDQ schemas are absent from standard shape inference.
        if node.op_type in {"QuantizeLinear", "DequantizeLinear", "Cast", "Identity", "Softmax"} and node.input[0] in shapes:
            for name in node.output:
                shapes[name] = shapes[node.input[0]]
    return shapes


def refine_dynamic_matmul_rhs(model, *, node_names=None, lhs_l1_bounds=None, expand_high_range=False):
    """Mutate a static ONNX QDQ model and return compact transform metadata.

    Only MatMul(U16-DQ, U8-DQ) where the U8 input is an explicit conversion
    from a dynamic U16-DQ tensor is eligible. Static weights and unrelated U8
    inputs are left untouched. Existing initializers, graph IO and final output
    quantizers are retained. Reapplying the transform is a no-op.

    node_names restricts the original MatMul names. lhs_l1_bounds optionally maps
    each name to a proven upper bound on sum(abs(A), axis=-1). Otherwise the bound
    is contraction_length * max(abs(A's representable endpoints)). Bounds affect
    both partial MatMul output encodings; incorrect tight bounds may clip.
    The default can be loose, so it is a diagnostic-safe starting point rather
    than an accuracy-optimal calibration recipe.

    expand_high_range replaces the high branch's encoding when needed to cover
    the source U16 range. This avoids spending residual precision on clipping
    tails of a narrower old U8 encoding. Original shared quantizers remain intact
    for other consumers. The high branch then differs from the original graph.
    """
    selected = None if node_names is None else set(node_names)
    lhs_l1_bounds = lhs_l1_bounds or {}
    initializers = {item.name: item for item in model.graph.initializer}
    producers = {output: node for node in model.graph.node for output in node.output}
    consumers = {}
    dynamic = {item.name: True for item in model.graph.input if item.name not in initializers}
    shapes = _static_shapes(model)
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
        for name in node.output:
            dynamic[name] = any(dynamic.get(input_name, False) for input_name in node.input)

    occupied = set(initializers) | set(producers) | {n.name for n in model.graph.node}
    new_initializers, new_info, new_nodes, records = [], [], [], []

    def unique(stem):
        name, suffix = stem, 0
        while name in occupied:
            suffix += 1
            name = f"{stem}_{suffix}"
        occupied.add(name)
        return name

    for index, node in enumerate(model.graph.node):
        if node.op_type != "MatMul" or (selected is not None and node.name not in selected):
            new_nodes.append(node)
            continue
        lhs_dq = producers.get(node.input[0])
        high_dq = producers.get(node.input[1])
        lhs_encoding = _scalar_encoding(lhs_dq, initializers, np.uint16)
        high_encoding = _scalar_encoding(high_dq, initializers, np.uint8)
        high_q = producers.get(high_dq.input[0]) if high_encoding else None
        rhs_dq = producers.get(high_q.input[0]) if high_q is not None and high_q.op_type == "QuantizeLinear" else None
        rhs_encoding = _scalar_encoding(rhs_dq, initializers, np.uint16)
        output_consumers = consumers.get(node.output[0], [])
        output_q = output_consumers[0] if len(output_consumers) == 1 else None
        output_encoding = _scalar_encoding(output_q, initializers, np.uint16)
        if (not lhs_encoding or not high_encoding or not rhs_encoding or not output_encoding
                or lhs_dq.op_type != "DequantizeLinear" or high_dq.op_type != "DequantizeLinear"
                or rhs_dq.op_type != "DequantizeLinear" or output_q.op_type != "QuantizeLinear"
                or _scalar_encoding(high_q, initializers, np.uint8) != high_encoding
                or not dynamic.get(rhs_dq.input[0], False)):
            new_nodes.append(node)
            continue
        # Transformed MatMuls consume intermediate encodings, not the original
        # conversion pattern. This marker also makes repeated application safe.
        if "__rhs_refine_" in node.name:
            new_nodes.append(node)
            continue
        lhs_shape, rhs_shape = shapes.get(node.input[0]), shapes.get(node.input[1])
        if lhs_shape is None or rhs_shape is None or len(lhs_shape) < 2 or len(rhs_shape) < 2:
            raise ValueError(f"{node.name}: refinement needs static inferred input shapes")
        if lhs_shape[-1] != rhs_shape[-2]:
            raise ValueError(f"{node.name}: incompatible MatMul contraction dimensions")
        output_shape = list(np.broadcast_shapes(tuple(lhs_shape[:-2]), tuple(rhs_shape[:-2]))) + [lhs_shape[-2], rhs_shape[-1]]
        contraction = rhs_shape[-2]
        lhs_min, lhs_max = _range(lhs_encoding, 65535)
        l1_bound = float(lhs_l1_bounds.get(node.name, contraction * max(abs(lhs_min), abs(lhs_max))))
        if not math.isfinite(l1_bound) or l1_bound <= 0:
            raise ValueError(f"{node.name}: lhs_l1_bound must be finite and positive")
        rhs_min, rhs_max = _range(rhs_encoding, 65535)
        existing_high_range = _range(high_encoding, 255)
        expanded = bool(expand_high_range and
                        (rhs_min < existing_high_range[0] or rhs_max > existing_high_range[1]))
        if expanded:
            high_scale, high_zero = _covering_encoding(rhs_min, rhs_max, 255)
            high_scale = float(high_scale)
        else:
            high_scale, high_zero = high_encoding
        high_min, high_max = _range((high_scale, high_zero), 255)
        # Cover normal nearest-rounding error AND saturation when the original
        # U16 range extends beyond the existing U8 range. Margin covers float32
        # arithmetic and the high-term U8 -> U16 requantization roundoff.
        margin = 8 * np.finfo(np.float32).eps * max(abs(rhs_min), abs(rhs_max), abs(high_min), abs(high_max), high_scale)
        residual_min = min(-high_scale / 2, rhs_min - high_min) - margin
        residual_max = max(high_scale / 2, rhs_max - high_max) + margin
        low_scale, low_zero = _covering_encoding(residual_min, residual_max, 255)
        low_extent = max(abs(x) for x in _range((float(low_scale), low_zero), 255))
        partial_scale, partial_zero = _covering_encoding(-l1_bound * low_extent, l1_bound * low_extent, 65535)
        partial_bound = max(abs(x) for x in _range((float(partial_scale), partial_zero), 65535))
        output_min, output_max = _range(output_encoding, 65535)
        # If |R| <= B, clipping H to [L-B,U+B] cannot change clipping
        # H+R to [L,U]. Reusing [L,U] for H would discard compensating R.
        high_output_scale, high_output_zero = _covering_encoding(output_min - partial_bound, output_max + partial_bound, 65535)
        prefix = (node.name or f"MatMul_{index}") + "__rhs_refine_"
        domain = rhs_dq.domain

        def initializer(role, value, dtype):
            name = unique(prefix + role)
            new_initializers.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
            return name

        def qdq(value, role, scale, zero, dtype, shape):
            s = initializer(role + "_scale", scale, np.float32)
            z = initializer(role + "_zero", zero, dtype)
            quantized, restored = unique(prefix + role + "_q"), unique(prefix + role + "_dq")
            new_nodes.extend([
                helper.make_node("QuantizeLinear", [value, s, z], [quantized], name=unique(prefix + role + "_Q"), domain=domain),
                helper.make_node("DequantizeLinear", [quantized, s, z], [restored], name=unique(prefix + role + "_DQ"), domain=domain),
            ])
            kind = TensorProto.UINT16 if dtype == np.uint16 else TensorProto.UINT8
            new_info.extend([helper.make_tensor_value_info(quantized, kind, shape), helper.make_tensor_value_info(restored, TensorProto.FLOAT, shape)])
            return restored

        # Every U8 code has an equivalent U16 code q16=q8*257 (up to
        # float32 scale representation). Sub receives two U16 operands.
        high_tensor = (qdq(high_q.input[0], "expanded_high_u8", high_scale, high_zero, np.uint8, rhs_shape)
                       if expanded else node.input[1])
        high_for_sub = qdq(high_tensor, "high_u16", high_scale / 257, high_zero * 257, np.uint16, rhs_shape)
        residual = unique(prefix + "residual")
        new_nodes.append(helper.make_node("Sub", [high_q.input[0], high_for_sub], [residual], name=unique(prefix + "Sub")))
        residual16 = qdq(residual, "residual_u16", low_scale / 257, low_zero * 257, np.uint16, rhs_shape)
        low = qdq(residual16, "low_u8", low_scale, low_zero, np.uint8, rhs_shape)

        high_output, low_output = unique(prefix + "high_output"), unique(prefix + "low_output")
        high_matmul = onnx.NodeProto()
        high_matmul.CopyFrom(node)
        high_matmul.name = unique(prefix + "high_MatMul")
        high_matmul.input[1] = high_tensor
        high_matmul.output[0] = high_output
        new_nodes.append(high_matmul)
        high_result = qdq(high_output, "high_output_u16", high_output_scale, high_output_zero, np.uint16, output_shape)
        new_nodes.append(helper.make_node("MatMul", [node.input[0], low], [low_output], name=unique(prefix + "low_MatMul")))
        low_result = qdq(low_output, "low_output_u16", partial_scale, partial_zero, np.uint16, output_shape)
        new_nodes.append(helper.make_node("Add", [high_result, low_result], list(node.output), name=unique(prefix + "Add")))
        records.append({
            "node": node.name, "rhs16_range": [float(rhs_min), float(rhs_max)],
            "high8_range": [float(high_min), float(high_max)],
            "existing_high8_range": list(map(float, existing_high_range)), "expanded_high_range": bool(expanded),
            "residual_range": [float(residual_min), float(residual_max)], "residual_scale": float(low_scale),
            "residual_zero_point": int(low_zero), "contraction_length": int(contraction),
            "lhs_l1_bound": float(l1_bound), "partial_output_scale": float(partial_scale),
            "partial_output_bound": float(partial_bound),
            "high_output_range": list(map(float, _range((float(high_output_scale), high_output_zero), 65535))),
        })

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_initializers)
    model.graph.value_info.extend(new_info)
    return {"refined_matmuls": len(records), "nodes": records}


def attention_matmul_refinement(model, scope="all"):
    """Refine eligible dynamic RHS conversions with a tight bound for P @ V.

    ``pv`` selects MatMuls whose left operand comes from Softmax through QDQ,
    float32 Cast and/or Identity nodes. ``all`` also refines other eligible
    dynamic MatMuls (including Q @ K) using their full representable L1 bound.
    Both scopes expand the coarse RHS encoding to cover its source U16 range.

    The PV bound is 1 + sequence_length * sum(rounding_scales) / 2 + 0.01.
    The rounding term assumes last-axis Softmax and a nonnegative U16 encoding
    whose maximum is at most one. Upper clipping only reduces its L1 norm.
    Each Q/DQ pair must agree; different successive encodings add rounding
    error, while adjacent equal encodings are idempotent and count only once.
    The 0.01 extra allowance is an empirical margin for backend arithmetic,
    not a mathematical guarantee for arbitrary hardware implementations.
    """
    if scope not in {"all", "pv"}:
        raise ValueError("scope must be 'all' or 'pv'")
    producers = {name: node for node in model.graph.node for name in node.output}
    initializers = {item.name: item for item in model.graph.initializer}
    shapes = _static_shapes(model)
    bounds, pv_records = {}, []
    expected_scale = float(np.float32(1 / 65535))
    for node in model.graph.node:
        if node.op_type != "MatMul" or "__rhs_refine_" in node.name:
            continue
        current, traversed, seen = node.input[0], [], set()
        while current not in seen:
            seen.add(current)
            source = producers.get(current)
            if source is None or source.op_type not in {"QuantizeLinear", "DequantizeLinear", "Cast", "Identity"}:
                break
            traversed.append(source)
            current = source.input[0]
        source = producers.get(current)
        if source is None or source.op_type != "Softmax":
            continue
        shape = shapes.get(node.input[0])
        if shape is None or len(shape) < 2:
            raise ValueError(f"{node.name}: PV bound requires a static Softmax shape")
        axis = next((int(attr.i) for attr in source.attribute if attr.name == "axis"), -1)
        if axis not in {-1, len(shape) - 1}:
            raise ValueError(f"{node.name}: Softmax must normalize the MatMul contraction axis")
        if not traversed or traversed[0].op_type != "DequantizeLinear":
            continue  # It is not the quantized MatMul pattern this helper refines.
        probability_encoding = None
        quantize_stages = []
        paired_quantizers = set()
        for intermediary in traversed:
            if intermediary.op_type in {"QuantizeLinear", "DequantizeLinear"}:
                encoding = _scalar_encoding(intermediary, initializers, np.uint16)
                if (encoding is None or encoding[1] != 0 or not np.isfinite(encoding[0])
                        or not 0 < encoding[0] <= expected_scale * (1 + 1e-6)):
                    raise ValueError(f"{node.name}: tight PV bound requires U16 0<scale<=1/65535 and zero_point=0")
                if probability_encoding is None:
                    probability_encoding = encoding  # Final MatMul input encoding.
                if intermediary.op_type == "DequantizeLinear":
                    quantizer = producers.get(intermediary.input[0])
                    if (quantizer is None or quantizer.op_type != "QuantizeLinear"
                            or _scalar_encoding(quantizer, initializers, np.uint16) != encoding):
                        raise ValueError(f"{node.name}: every probability Q/DQ pair must have matching encodings")
                    paired_quantizers.add(quantizer.output[0])
                    quantize_stages.append(encoding)
            elif intermediary.op_type == "Cast":
                target = next((int(attr.i) for attr in intermediary.attribute if attr.name == "to"), None)
                if target != TensorProto.FLOAT:
                    raise ValueError(f"{node.name}: tight PV bound only supports float32 Cast intermediaries")
        if any(item.output[0] not in paired_quantizers for item in traversed
               if item.op_type == "QuantizeLinear"):
            raise ValueError(f"{node.name}: tight PV bound requires paired probability quantizers")
        rounding_scales = []
        previous = None
        for encoding in reversed(quantize_stages):
            if encoding != previous:
                rounding_scales.append(float(encoding[0]))
            previous = encoding
        length = int(shape[-1])
        probability_scale = float(probability_encoding[0])
        bounds[node.name] = 1.0 + length * sum(rounding_scales) / 2 + 0.01
        pv_records.append({"node": node.name, "softmax": source.name,
                           "sequence_length": length, "probability_scale": probability_scale,
                           "probability_quantize_stages": len(quantize_stages),
                           "probability_rounding_scales": rounding_scales,
                           "lhs_l1_bound": bounds[node.name]})
    result = refine_dynamic_matmul_rhs(
        model, node_names=set(bounds) if scope == "pv" else None,
        lhs_l1_bounds=bounds, expand_high_range=True,
    )
    refined_names = {item["node"] for item in result["nodes"]}
    result.update({
        "scope": scope,
        "pv_bounds": [record for record in pv_records if record["node"] in refined_names],
        "assumptions": {
            "pv_l1_formula": "1 + sequence_length * sum(probability_rounding_scales) / 2 + 0.01",
            "pv_quantization": "Last-axis Softmax; matching U16 Q/DQ pairs, 0<scale<=1/65535, zero_point=0; nearest rounding; adjacent equal stages counted once; upper clipping cannot increase L1",
            "backend_l1_margin": 0.01,
            "backend_margin_status": "Empirical allowance; not a universal hardware error guarantee",
            "other_l1_bound": "contraction_length * max(abs(lhs representable endpoints))",
            "rhs_high_range": "Covers the source U16 representable range",
        },
    })
    return result
