"""Small synthetic CPU/optional strict-HTP probe of U16 LayerNorm affines.

Run `python -m npu.probe_affine_layernorm --out NEW_DIRECTORY --htp` only with
exclusive access to the Pi accelerator. No task/evaluation examples are used.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh

from npu.affine_layernorm import refine_layernorm_affines
from npu.matmul_refinement import _covering_encoding


def make_probe_models(*, channels=768, tokens=64, nonzero_beta=False):
    """Return a QDQ graph and reference differing only in affine precision."""
    gamma = np.linspace(.638, 3.174, channels, dtype=np.float32)
    gamma[:2] = [-.00609, 0]
    beta = np.linspace(-.137, .219, channels, dtype=np.float32) if nonzero_beta else np.zeros(channels, np.float32)
    xs, xz = _covering_encoding(-8, 8, 65535)
    ys, yz = _covering_encoding(-20, 20, 65535)
    gs, gz = _covering_encoding(float(gamma.min()), float(gamma.max()), 255)
    gq = np.clip(np.rint(gamma / gs) + gz, 0, 255).astype(np.uint8)
    bs = np.float32(xs * gs)
    bq = np.rint(beta / bs).astype(np.int32)
    values = {
        "xs": np.asarray(xs), "xz": np.asarray(xz, np.uint16),
        "ys": np.asarray(ys), "yz": np.asarray(yz, np.uint16),
        "gq": gq, "gs": np.asarray(gs), "gz": np.asarray(gz, np.uint8),
        "bq": bq, "bs": np.asarray([bs]), "bz": np.asarray(0, np.int32),
    }
    nodes = [
        h.make_node("QuantizeLinear", ["x", "xs", "xz"], ["xq"], name="xQ"),
        h.make_node("DequantizeLinear", ["xq", "xs", "xz"], ["xdq"], name="xDQ"),
        h.make_node("DequantizeLinear", ["gq", "gs", "gz"], ["gamma"], name="gammaDQ"),
        h.make_node("DequantizeLinear", ["bq", "bs", "bz"], ["beta"], name="betaDQ"),
        h.make_node("LayerNormalization", ["xdq", "gamma", "beta"], ["raw"], name="norm", axis=-1, epsilon=1e-5),
        h.make_node("QuantizeLinear", ["raw", "ys", "yz"], ["yq"], name="yQ"),
        h.make_node("DequantizeLinear", ["yq", "ys", "yz"], ["y"], name="yDQ"),
    ]
    shape = [1, tokens, channels]
    graph = h.make_graph(nodes, "affine_ln_probe", [h.make_tensor_value_info("x", T.FLOAT, shape)],
                         [h.make_tensor_value_info("y", T.FLOAT, shape)],
                         [nh.from_array(value, name) for name, value in values.items()],
                         value_info=[h.make_tensor_value_info("raw", T.FLOAT, shape)])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 21)], ir_version=10)
    reference = copy.deepcopy(model)
    keep = [node for node in reference.graph.node if node.name not in {"gammaDQ", "betaDQ"}]
    del reference.graph.node[:]
    reference.graph.node.extend(keep)
    reference.graph.initializer.extend([nh.from_array(gamma, "gamma"), nh.from_array(beta, "beta")])
    return model, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--htp", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    from npu.calibrate_conv_offsets import runtime_session
    if args.htp:
        from laya_npu import _qnn_runtime
        _qnn_runtime()
    rng = np.random.default_rng(607)
    inputs = [rng.normal(size=(1, 64, 768)).astype(np.float32), np.zeros((1, 64, 768), np.float32)]
    result = {"synthetic_seed": 607, "strict_htp_requested": args.htp, "probes": []}
    for with_beta in (False, True):
        model, reference = make_probe_models(nonzero_beta=with_beta)
        refined = copy.deepcopy(model)
        metadata = refine_layernorm_affines(refined, reference)
        outputs = {}
        prefix = "beta" if with_beta else "gamma"
        for label, graph in (("reference", reference), ("original", model), ("refined", refined)):
            onnx.checker.check_model(graph)
            path = args.out / f"{prefix}_{label}.onnx"
            onnx.save(graph, path)
            session = runtime_session(path, False)
            outputs[label] = [session.run(None, {"x": value})[0] for value in inputs]
            del session
        if args.htp:
            session = runtime_session(args.out / f"{prefix}_refined.onnx", True)
            session.disable_fallback()
            outputs["htp"] = [session.run(None, {"x": value})[0] for value in inputs]
            del session
        records = []
        for index in range(len(inputs)):
            errors = {}
            for label, values in outputs.items():
                if label == "reference":
                    continue
                delta = np.abs(values[index] - outputs["reference"][index])
                errors[label + "_vs_reference"] = {"mean_abs": float(delta.mean()), "max_abs": float(delta.max())}
            if "htp" in outputs:
                delta = np.abs(outputs["htp"][index] - outputs["refined"][index])
                errors["htp_vs_refined_cpu"] = {"mean_abs": float(delta.mean()), "max_abs": float(delta.max())}
            records.append(errors)
        result["probes"].append({"nonzero_beta": with_beta, "metadata": metadata, "inputs": records})
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
