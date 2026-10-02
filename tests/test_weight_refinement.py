"""CPU numerical and graph-contract tests for two-term Conv weights."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T, helper as h, numpy_helper as nh

from npu.calibrate_conv_offsets import conv_records
from npu.weight_refinement import refine_conv_weights, refine_files
from npu.fidelity_runtime import sha256_file


def tiny_model(weights, scales, *, output_scale=20 / 65535, bias=None):
    weights, scales = np.asarray(weights, np.float32), np.asarray(scales, np.float32)
    shape = [1, weights.shape[1], 1, 5]
    output_shape = [1, weights.shape[0], 1, 5]
    arrays = {
        "xs": np.array(2 / 65535, np.float32), "xz": np.array(32768, np.uint16),
        "wq": np.clip(np.rint(weights / scales[:, None, None, None]), -127, 127).astype(np.int8),
        "ws": scales, "wz": np.zeros(weights.shape[0], np.int8),
        "ys": np.array(output_scale, np.float32), "yz": np.array(32768, np.uint16),
    }
    nodes = [h.make_node("QuantizeLinear", ["x", "xs", "xz"], ["xq"], name="xQ", domain="com.microsoft"),
             h.make_node("DequantizeLinear", ["xq", "xs", "xz"], ["xdq"], name="xDQ", domain="com.microsoft"),
             h.make_node("DequantizeLinear", ["wq", "ws", "wz"], ["wdq"], name="wDQ", axis=0)]
    inputs = ["xdq", "wdq"]
    if bias is not None:
        arrays["bias"] = np.asarray(bias, np.float32)
        inputs.append("bias")
    nodes += [h.make_node("Conv", inputs, ["y"], name="projection", kernel_shape=[1, 1]),
              h.make_node("QuantizeLinear", ["y", "ys", "yz"], ["yq"], name="yQ", domain="com.microsoft"),
              h.make_node("DequantizeLinear", ["yq", "ys", "yz"], ["ydq"], name="yDQ", domain="com.microsoft")]
    model = h.make_model(h.make_graph(nodes, "weight_test", [h.make_tensor_value_info("x", T.FLOAT, shape)],
                                     [h.make_tensor_value_info("ydq", T.FLOAT, output_shape)],
                                     [nh.from_array(value, name) for name, value in arrays.items()]),
                        opset_imports=[h.make_opsetid("", 21), h.make_opsetid("com.microsoft", 1)], ir_version=10)
    model = onnx.shape_inference.infer_shapes(model)
    source = h.make_model(h.make_graph(
        [h.make_node("Conv", ["x", "weights"], ["y"], name="projection", kernel_shape=[1, 1])], "fp32",
        [h.make_tensor_value_info("x", T.FLOAT, shape)], [h.make_tensor_value_info("y", T.FLOAT, output_shape)],
        [nh.from_array(weights, "weights")]), opset_imports=[h.make_opsetid("", 21)], ir_version=10)
    return model, source


def run(model, x):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]


def reference(weights, x, output_scale=20 / 65535, bias=None):
    xs = np.float32(2 / 65535)
    xdq = (np.clip(np.rint(x / xs) + 32768, 0, 65535) - 32768) * xs
    y = np.einsum("oc,nchw->nohw", weights[:, :, 0, 0], xdq)
    if bias is not None:
        y += np.asarray(bias, np.float32)[None, :, None, None]
    ys = np.float32(output_scale)
    return (np.clip(np.rint(y / ys) + 32768, 0, 65535) - 32768) * ys


class WeightRefinementTests(unittest.TestCase):
    def test_reduces_weight_and_output_error_preserves_interface_and_bias(self):
        rng = np.random.default_rng(192)
        scales = np.linspace(.02, .05, 7, dtype=np.float32)
        weights = ((rng.integers(-20, 20, (7, 64, 1, 1)) + .49) * scales[:, None, None, None]).astype(np.float32)
        x = rng.uniform(-.9, .9, (1, 64, 1, 5)).astype(np.float32)
        bias = rng.normal(0, .1, 7).astype(np.float32)
        model, source = tiny_model(weights, scales, bias=bias)
        before = run(model, x)
        original_io = ([item.SerializeToString() for item in model.graph.input],
                       [item.SerializeToString() for item in model.graph.output])
        original_initializers = {item.name: item.SerializeToString() for item in model.graph.initializer}
        metadata = refine_conv_weights(model, source)
        onnx.checker.check_model(model)
        after = run(model, x)
        expected = reference(weights, x, bias=bias)
        self.assertLess(np.linalg.norm(after - expected), np.linalg.norm(before - expected) * .03)
        self.assertEqual(metadata["refined_convs"], 1)
        self.assertLess(metadata["nodes"][0]["weight_max_error_after"], metadata["nodes"][0]["weight_max_error_before"] * .005)
        self.assertEqual(metadata, json.loads(json.dumps(metadata, allow_nan=False)))
        self.assertEqual(original_io, ([item.SerializeToString() for item in model.graph.input],
                                       [item.SerializeToString() for item in model.graph.output]))
        for item in model.graph.initializer:
            if item.name in original_initializers:
                self.assertEqual(item.SerializeToString(), original_initializers[item.name])
        convs = [node for node in model.graph.node if node.op_type == "Conv"]
        self.assertEqual(list(convs[0].input)[-1], "bias")
        self.assertEqual(len(convs[1].input), 2)
        records = conv_records(model)
        self.assertEqual(len(records), 2)  # Both are compatible with fresh HTP offset calibration.
        unchanged = model.SerializeToString()
        self.assertEqual(refine_conv_weights(model, source)["refined_convs"], 0)
        self.assertEqual(unchanged, model.SerializeToString())

    def test_high_branch_range_keeps_cancellation_at_both_clipping_limits(self):
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                weights = np.full((1, 101, 1, 1), sign * .00986, np.float32)
                x = np.ones((1, 101, 1, 5), np.float32)
                model, source = tiny_model(weights, [.01], output_scale=2 / 65535)
                before = run(model, x)
                metadata = refine_conv_weights(model, source)
                expected = reference(weights, x, output_scale=2 / 65535)
                after = run(model, x)
                self.assertLess(float(np.abs(after - expected).max()), 8e-5)
                self.assertGreater(float(np.abs(before - expected).max()), .003)
                self.assertGreater(metadata["nodes"][0]["high_output_range"][1], 1.01)
                self.assertLess(metadata["nodes"][0]["high_output_range"][0], -1.01)

    def test_mixed_residuals_zero_rows_and_valid_bound(self):
        rng = np.random.default_rng(71)
        weights = rng.normal(0, .03, (3, 12, 1, 1)).astype(np.float32)
        weights[0] = 0
        model, source = tiny_model(weights, [.001, .001, .001])
        metadata = refine_conv_weights(model, source, activation_l1_bounds={"projection": 3.0})
        onnx.checker.check_model(model)
        arrays = {item.name: nh.to_array(item) for item in model.graph.initializer}
        low_w = arrays["projection__weight_refine_weight_q"]
        self.assertTrue(np.all(low_w[0] == 0))
        self.assertTrue(np.isfinite(arrays["projection__weight_refine_weight_scale"]).all())
        x = rng.uniform(-.2, .2, (1, 12, 1, 5)).astype(np.float32)
        self.assertTrue(np.isfinite(run(model, x)).all())
        self.assertEqual(metadata["nodes"][0]["activation_l1_bound"], 3.0)

    def test_selection_exact_weights_and_bad_source_are_safe(self):
        weights = np.full((2, 3, 1, 1), .0149, np.float32)
        model, source = tiny_model(weights, [.01, .01])
        original = model.SerializeToString()
        self.assertEqual(refine_conv_weights(model, source, node_names=[])["refined_convs"], 0)
        self.assertEqual(original, model.SerializeToString())
        bad_source = copy.deepcopy(source)
        bad_source.graph.node[0].name = "unrelated"
        with self.assertRaisesRegex(ValueError, "No matching"):
            refine_conv_weights(model, bad_source)
        self.assertEqual(original, model.SerializeToString())
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            refine_conv_weights(model, source, activation_l1_bounds={"projection": float("nan")})
        self.assertEqual(original, model.SerializeToString())
        exact_model, exact_source = tiny_model(np.full((2, 3, 1, 1), np.float32(.01)), [.01, .01])
        original_exact = exact_model.SerializeToString()
        self.assertEqual(refine_conv_weights(exact_model, exact_source)["refined_convs"], 0)
        self.assertEqual(original_exact, exact_model.SerializeToString())

    def test_resolves_shared_fp32_weight_alias(self):
        weights = np.full((2, 3, 1, 1), .0149, np.float32)
        model, source = tiny_model(weights, [.01, .01])
        source.graph.node[0].input[1] = "alias"
        source.graph.node.insert(0, h.make_node("Identity", ["weights"], ["alias"]))
        self.assertEqual(refine_conv_weights(model, source)["refined_convs"], 1)

    def test_shared_projections_share_residual_weight_storage(self):
        weights = np.full((2, 3, 1, 1), .0149, np.float32)
        model, source = tiny_model(weights, [.01, .01])
        model.graph.node.extend([
            h.make_node("Conv", ["xdq", "wdq"], ["y2"], name="projection2", kernel_shape=[1, 1]),
            h.make_node("QuantizeLinear", ["y2", "ys", "yz"], ["yq2"], name="yQ2", domain="com.microsoft"),
            h.make_node("DequantizeLinear", ["yq2", "ys", "yz"], ["ydq2"], name="yDQ2", domain="com.microsoft"),
        ])
        model.graph.output.append(h.make_tensor_value_info("ydq2", T.FLOAT, [1, 2, 1, 5]))
        source.graph.node.append(h.make_node("Conv", ["x", "weights"], ["y2"], name="projection2", kernel_shape=[1, 1]))
        metadata = refine_conv_weights(model, source)
        onnx.checker.check_model(model)
        self.assertEqual(metadata["refined_convs"], 2)
        self.assertEqual(metadata["residual_weight_tensors"], 1)
        low_convs = [node for node in model.graph.node if node.op_type == "Conv" and "low_Conv" in node.name]
        self.assertEqual(low_convs[0].input[1], low_convs[1].input[1])
        self.assertTrue(np.isfinite(run(model, np.ones((1, 3, 1, 5), np.float32))).all())

    def test_rejects_different_parameters_inside_matching_shape_and_names(self):
        weights = np.full((2, 3, 1, 1), .0149, np.float32)
        model, source = tiny_model(weights, [.01, .01])
        source.graph.initializer[0].CopyFrom(nh.from_array(weights + .001, "weights"))
        original = model.SerializeToString()
        with self.assertRaisesRegex(ValueError, "nearest-rounding cells"):
            refine_conv_weights(model, source)
        self.assertEqual(original, model.SerializeToString())


class WeightRefinementFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        model, reference_model = tiny_model(np.full((2, 3, 1, 1), .0149, np.float32), [.01, .01])
        self.source, self.fp32 = self.root / "source.onnx", self.root / "fp32.onnx"
        onnx.save(model, self.source)
        onnx.save(reference_model, self.fp32)
        self.checkpoint = self.root / "checkpoint.safetensors"
        self.checkpoint.write_bytes(b"synthetic checkpoint identity fixture")
        checkpoint_hash = sha256_file(self.checkpoint)
        self.manifest = {"checkpoint_sha256": checkpoint_hash, "mask_penalty": -100,
                         "zero_pad_embeddings": True, "precision": "a16w8",
                         "buckets": {"5": "source.onnx"}, "model_sha256": {"5": sha256_file(self.source)}}
        self.sidecar = {"fp32_sha256": sha256_file(self.fp32), "config": {
            "checkpoint_sha256": checkpoint_hash, "length": 5, "conv_linear": True,
            "mask_penalty": -100, "zero_pad_embeddings": True,
            "input_sha256": {"calibration_dataset_sha256": "1" * 64,
                             "model_files_sha256": {name: "2" * 64 for name in
                             ("encoder/config.json", "rl_agent_config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json")}}}}
        self.manifest_path = self.root / "manifest.json"
        self.out = self.root / "candidate"
        self.write_metadata()

    def write_metadata(self):
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.fp32.with_suffix(".json").write_text(json.dumps(self.sidecar))

    def build(self):
        return refine_files(source=self.source, fp32=self.fp32, source_manifest=self.manifest_path,
                            out_dir=self.out, checkpoint=self.checkpoint)

    def test_complete_artifacts_and_immutable_output(self):
        source_before = self.source.read_bytes()
        result = self.build()
        self.assertTrue(result["checkpoint_bytes_verified"])
        self.assertFalse(result["source_input_hashes_verified_against_files"])
        saved_manifest = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual(saved_manifest["model_sha256"]["5"], sha256_file(self.out / "source.onnx"))
        self.assertEqual(saved_manifest["weight_refinement"], "two_term_int8")
        onnx.checker.check_model(onnx.load(self.out / "source.onnx"))
        self.assertEqual(self.source.read_bytes(), source_before)
        original_result = (self.out / "weight-refinement.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual((self.out / "weight-refinement.json").read_bytes(), original_result)

    def test_checksum_checkpoint_and_input_contract_fail_before_writes(self):
        for label, alter, pattern in (
            ("source", lambda: self.manifest["model_sha256"].update({"5": "0" * 64}), "Source manifest SHA256"),
            ("fp32", lambda: self.sidecar.update({"fp32_sha256": "0" * 64}), "sidecar SHA256"),
            ("checkpoint", lambda: self.sidecar["config"].update({"checkpoint_sha256": "0" * 64}), "checkpoint hashes differ"),
            ("bucket", lambda: self.sidecar["config"].update({"length": 7}), "same bucket"),
            ("padding", lambda: self.sidecar["config"].update({"zero_pad_embeddings": False}), "zero_pad_embeddings differ"),
            ("input provenance", lambda: self.sidecar["config"].pop("input_sha256"), "lacks input hashes"),
        ):
            with self.subTest(label=label):
                old_manifest, old_sidecar = copy.deepcopy(self.manifest), copy.deepcopy(self.sidecar)
                alter()
                self.write_metadata()
                with self.assertRaisesRegex(ValueError, pattern):
                    self.build()
                self.assertFalse(self.out.exists())
                self.manifest, self.sidecar = old_manifest, old_sidecar
        self.write_metadata()
        self.checkpoint.write_bytes(b"different checkpoint")
        with self.assertRaisesRegex(ValueError, "Checkpoint file SHA256"):
            self.build()
        self.assertFalse(self.out.exists())

    def test_missing_sidecar_or_hidden_htp_correction_rejected(self):
        self.fp32.with_suffix(".json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.build()
        self.assertFalse(self.out.exists())
        self.write_metadata()
        model = onnx.load(self.source)
        model.graph.initializer.append(nh.from_array(np.array([1], np.int32), "__htp_offset_conv_000_q"))
        onnx.save(model, self.source)
        self.manifest["model_sha256"]["5"] = sha256_file(self.source)
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "already contains HTP"):
            self.build()
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
