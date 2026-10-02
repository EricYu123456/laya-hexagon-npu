"""Numerical and graph-contract checks for explicit U16 LayerNorm affines."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper as nh

from npu.affine_layernorm import refine_files, refine_layernorm_affines
from npu.fidelity_runtime import sha256_file
from npu.probe_affine_layernorm import make_probe_models


def evaluate(model, values):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    return session.run(None, {"x": values})[0]


class AffineLayerNormTests(unittest.TestCase):
    def test_reduces_gamma_error_using_original_fp32_parameters(self):
        model, reference = make_probe_models()
        original = copy.deepcopy(model)
        io = ([item.SerializeToString() for item in model.graph.input], [item.SerializeToString() for item in model.graph.output])
        result = refine_layernorm_affines(model, reference)
        self.assertEqual(result["refined_layer_norms"], 1)
        self.assertLess(result["nodes"][0]["gamma_max_abs_error"], 3e-5)
        json.dumps(result, allow_nan=False)
        onnx.checker.check_model(model)
        x = np.random.default_rng(610).normal(size=(1, 64, 768)).astype(np.float32)
        ideal, before, after = (evaluate(graph, x) for graph in (reference, original, model))
        self.assertLess(np.abs(after - ideal).mean(), np.abs(before - ideal).mean() / 4)
        self.assertLess(np.abs(after - ideal).max(), .003)
        self.assertEqual(io, ([item.SerializeToString() for item in model.graph.input], [item.SerializeToString() for item in model.graph.output]))
        encoded = {item.name: nh.to_array(item) for item in model.graph.initializer}
        self.assertEqual(encoded["norm__affine16_gamma_q"].dtype, np.uint16)
        self.assertEqual(encoded["norm__affine16_unit_gamma_q"].dtype, np.uint8)
        self.assertTrue(np.all(encoded["norm__affine16_unit_gamma_q"].astype(np.float32) * encoded["norm__affine16_unit_gamma_scale"] == 1))
        after_once = model.SerializeToString()
        self.assertEqual(refine_layernorm_affines(model, reference)["refined_layer_norms"], 0)
        self.assertEqual(model.SerializeToString(), after_once)

    def test_nonzero_beta_and_constant_inputs(self):
        model, reference = make_probe_models(nonzero_beta=True)
        original = copy.deepcopy(model)
        result = refine_layernorm_affines(model, reference)
        self.assertTrue(result["nodes"][0]["nonzero_beta"])
        for x in (np.zeros((1, 64, 768), np.float32), np.ones((1, 64, 768), np.float32),
                  np.random.default_rng(1).normal(size=(1, 64, 768)).astype(np.float32)):
            ideal, before, after = (evaluate(graph, x) for graph in (reference, original, model))
            self.assertLess(np.abs(after - ideal).max(), .004)
            if x.std():
                self.assertLess(np.abs(after - ideal).mean(), np.abs(before - ideal).mean() / 2)

    def test_invalid_reference_or_input_rejected_without_mutation(self):
        for corruption in ("name", "axis", "dynamic", "activation"):
            model, reference = make_probe_models()
            norm = next(node for node in reference.graph.node if node.op_type == "LayerNormalization")
            if corruption == "name":
                norm.name = "different"
            elif corruption == "axis":
                next(item for item in norm.attribute if item.name == "axis").i = 1
            elif corruption == "dynamic":
                norm.input[1] = "x"
            else:
                next(node for node in model.graph.node if node.op_type == "LayerNormalization").input[0] = "x"
            before = model.SerializeToString()
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                refine_layernorm_affines(model, reference)
            self.assertEqual(model.SerializeToString(), before)

    def test_selection_leaves_other_nodes_unchanged(self):
        model, reference = make_probe_models()
        before = model.SerializeToString()
        self.assertEqual(refine_layernorm_affines(model, reference, node_names={"other"})["refined_layer_norms"], 0)
        self.assertEqual(model.SerializeToString(), before)

    def test_near_constant_and_unit_bound_inputs(self):
        model, reference = make_probe_models()
        refine_layernorm_affines(model, reference)
        impulse = np.zeros((1, 64, 768), np.float32)
        impulse[:, :, 2] = 7  # Unit LN approaches sqrt(767), but affine output fits.
        for x in (impulse, -impulse, np.ones_like(impulse) + impulse * 1e-4):
            ideal, actual = (evaluate(graph, x) for graph in (reference, model))
            self.assertTrue(np.isfinite(actual).all())
            self.assertLess(np.abs(actual - ideal).max(), .003)

    def test_ambiguous_names_or_static_dimension_rejected(self):
        for issue in ("duplicate", "empty", "dimension"):
            model, reference = make_probe_models()
            norm = next(node for node in model.graph.node if node.op_type == "LayerNormalization")
            if issue == "duplicate":
                model.graph.node.append(copy.deepcopy(norm))
            elif issue == "empty":
                norm.name = ""
            else:
                model.graph.value_info[0].type.tensor_type.shape.dim[-1].dim_value = 767
            before = model.SerializeToString()
            with self.subTest(issue=issue), self.assertRaises(ValueError):
                refine_layernorm_affines(model, reference)
            self.assertEqual(model.SerializeToString(), before)


class AffineLayerNormFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        build, weight = self.root / "build", self.root / "weight"
        build.mkdir()
        weight.mkdir()
        self.source, self.fp32 = weight / "backbone_3_a16w8.onnx", build / "backbone_3_fp32.onnx"
        source_model, reference = make_probe_models(channels=8, tokens=3)
        onnx.save(source_model, self.source)
        onnx.save(reference, self.fp32)
        self.expected = copy.deepcopy(source_model)
        refine_layernorm_affines(self.expected, reference)
        self.checkpoint = self.root / "checkpoint.safetensors"
        self.checkpoint.write_bytes(b"synthetic model identity")
        checkpoint_hash = sha256_file(self.checkpoint)
        self.sidecar = {"fp32_sha256": sha256_file(self.fp32), "config": {
            "checkpoint_sha256": checkpoint_hash, "length": 3, "conv_linear": True,
            "mask_penalty": -100, "zero_pad_embeddings": True,
            "input_sha256": {"calibration_dataset_sha256": "1" * 64, "model_files_sha256": {
                name: "2" * 64 for name in ("encoder/config.json", "rl_agent_config.json",
                                           "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json")}}}}
        self.sidecar_path = self.fp32.with_suffix(".json")
        self.sidecar_path.write_text(json.dumps(self.sidecar))
        self.manifest = {"checkpoint_sha256": checkpoint_hash, "mask_penalty": -100,
                         "zero_pad_embeddings": True, "precision": "a16w8", "weight_refinement": "two_term_int8",
                         "buckets": {"3": self.source.name}, "model_sha256": {"3": sha256_file(self.source)}}
        self.manifest_path = weight / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest))
        build_manifest = {**self.manifest, "model_sha256": {"3": "3" * 64}}
        (build / "manifest.json").write_text(json.dumps(build_manifest))
        (build / "backbone_3_a16w8.json").write_text(json.dumps({"model_sha256": "3" * 64, "feature_group_plan": {"groups": [[0, 1]]}}))
        self.weight_record = {
            "target_sha256": sha256_file(self.source), "source_qdq_sha256": "3" * 64,
            "source_fp32_sha256": sha256_file(self.fp32), "fp32_sidecar_sha256": sha256_file(self.sidecar_path),
            "checkpoint_sha256": checkpoint_hash, "fp32_export_config": self.sidecar["config"],
            "source_manifest_sha256": sha256_file(build / "manifest.json"),
        }
        self.weight_path = weight / "weight-refinement.json"
        self.weight_path.write_text(json.dumps(self.weight_record))
        self.out = self.root / "candidate"

    def build(self, **kwargs):
        return refine_files(source=self.source, fp32=self.fp32, source_manifest=self.manifest_path,
                            out_dir=self.out, checkpoint=self.checkpoint, expected_layernorms=1, **kwargs)

    def test_matches_library_bytes_preserves_metadata_and_output_is_immutable(self):
        original_source = self.source.read_bytes()
        result = self.build()
        self.assertEqual((self.out / self.source.name).read_bytes(), self.expected.SerializeToString())
        self.assertEqual(self.source.read_bytes(), original_source)
        self.assertTrue(result["checkpoint_bytes_verified"])
        self.assertFalse(result["source_input_hashes_verified_against_files"])
        self.assertEqual(set(result["retained_metadata_sha256"]), {
            "weight-refinement.json", "fp32-export.json", "build-metadata.json",
            "source-build-manifest.json", "source-weight-manifest.json"})
        self.assertEqual((self.out / "weight-refinement.json").read_bytes(), self.weight_path.read_bytes())
        saved = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual(saved["model_sha256"]["3"], result["target_sha256"])
        self.assertEqual(saved["layernorm_affine_refinement"], "unit_ln_u16_affine")
        original_result = (self.out / "affine-refinement.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual((self.out / "affine-refinement.json").read_bytes(), original_result)

    def test_source_and_provenance_mismatches_fail_before_writes(self):
        for field in ("target_sha256", "source_fp32_sha256", "fp32_sidecar_sha256", "checkpoint_sha256"):
            with self.subTest(field=field):
                changed = {**self.weight_record, field: "0" * 64}
                self.weight_path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "Weight provenance"):
                    self.build()
                self.assertFalse(self.out.exists())
        self.weight_path.write_text(json.dumps(self.weight_record))
        self.manifest["model_sha256"]["3"] = "0" * 64
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "Source manifest SHA256"):
            self.build()
        self.assertFalse(self.out.exists())

    def test_checkpoint_and_build_metadata_checks_precede_publication(self):
        self.checkpoint.write_bytes(b"different checkpoint")
        with self.assertRaisesRegex(ValueError, "Checkpoint file SHA256"):
            self.build()
        self.assertFalse(self.out.exists())
        self.checkpoint.write_bytes(b"synthetic model identity")
        build_path = self.fp32.parent / "backbone_3_a16w8.json"
        build_path.write_text('{"model_sha256":"wrong"}')
        with self.assertRaisesRegex(ValueError, "Build metadata"):
            self.build()
        self.assertFalse(self.out.exists())
        with self.assertRaises(FileNotFoundError):
            self.build(build_metadata=self.root / "missing.json")
        self.assertFalse(self.out.exists())

    def test_hidden_correction_and_already_refined_source_are_rejected(self):
        for kind in ("correction", "refined"):
            with self.subTest(kind=kind):
                model, _ = make_probe_models(channels=8, tokens=3)
                if kind == "correction":
                    model.graph.initializer.append(nh.from_array(np.array([0], np.int32), "__htp_offset_fake_q"))
                else:
                    model = self.expected
                onnx.save(model, self.source)
                digest = sha256_file(self.source)
                self.manifest["model_sha256"]["3"] = digest
                self.weight_record["target_sha256"] = digest
                self.manifest_path.write_text(json.dumps(self.manifest))
                self.weight_path.write_text(json.dumps(self.weight_record))
                with self.assertRaisesRegex(ValueError, "Source graph already contains"):
                    self.build()
                self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
