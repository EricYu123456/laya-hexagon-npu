"""Build reuse/publication guards with file fixtures and no model libraries."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tests.test_long_calibration import import_builder_without_model_libraries


class BuildPreflightTests(unittest.TestCase):
    def setUp(self):
        self.builder = import_builder_without_model_libraries()
        self.temporary = tempfile.TemporaryDirectory(prefix="laya-build-preflight-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model"
        for name in ("model.safetensors", "rl_agent_config.json", "encoder/config.json",
                     "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"):
            path = self.model / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
        self.dataset = self.root / "data.parquet"
        self.dataset.write_bytes(b"reserved calibration data")
        self.output = self.root / "build"
        self.output.mkdir()
        self.fp32 = self.output / "backbone_1024_fp32.onnx"
        self.qdq = self.output / "backbone_1024_a16w8.onnx"
        self.args = SimpleNamespace(output_dir=self.output, length=1024, mask_penalty=-100,
                                    activation_bits=16, zero_pad_embeddings=True,
                                    refine_matmul_rhs="all", reuse_export=False)
        self.config = {"checkpoint_sha256": self.builder.sha256_file(self.model / "model.safetensors"),
                       "input_sha256": self.builder.export_input_hashes(self.model, self.dataset)}

    def preflight(self, config=None):
        return self.builder.preflight_build(self.args, self.fp32, self.qdq, config or self.config)

    def save_export(self):
        self.fp32.write_bytes(b"verified FP32 export")
        self.fp32.with_suffix(".json").write_text(json.dumps({
            "config": self.config, "fp32_sha256": self.builder.sha256_file(self.fp32),
        }))
        self.args.reuse_export = True

    def test_reuse_accepts_verified_export_and_export_only_metadata_without_writes(self):
        self.save_export()
        metadata = self.qdq.with_suffix(".json")
        metadata.write_text("metadata from export-only phase")
        before = {path.name: path.read_bytes() for path in self.output.iterdir()}
        manifest = self.preflight()
        self.assertEqual(manifest["buckets"], {})
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.output.iterdir()})

    def test_changed_calibration_and_all_tokenizer_config_inputs_reject_stale_export(self):
        self.save_export()
        for path in (self.dataset, self.model / "rl_agent_config.json", self.model / "encoder/config.json",
                     self.model / "tokenizer/tokenizer.json", self.model / "tokenizer/tokenizer_config.json"):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                changed = {**self.config, "input_sha256": self.builder.export_input_hashes(self.model, self.dataset)}
                with self.assertRaisesRegex(ValueError, "input hashes"):
                    self.preflight(changed)
                path.write_bytes(original)
        (self.model / "tokenizer/added_tokens.json").write_text('{"new": 1}')
        changed = {**self.config, "input_sha256": self.builder.export_input_hashes(self.model, self.dataset)}
        with self.assertRaisesRegex(ValueError, "input hashes"):
            self.preflight(changed)

    def test_legacy_sidecars_and_changed_fp32_graph_fail_closed(self):
        self.save_export()
        sidecar = self.fp32.with_suffix(".json")
        saved = json.loads(sidecar.read_text())
        legacy = copy.deepcopy(saved)
        del legacy["config"]["input_sha256"]
        sidecar.write_text(json.dumps(legacy))
        with self.assertRaisesRegex(ValueError, "Legacy.*fresh export"):
            self.preflight()
        sidecar.write_text(json.dumps(saved))
        self.fp32.write_bytes(b"modified export")
        with self.assertRaisesRegex(ValueError, "Existing export"):
            self.preflight()

    def test_existing_quantized_graph_is_preserved_even_with_reuse(self):
        self.save_export()
        self.qdq.write_bytes(b"frozen quantized model")
        with self.assertRaisesRegex(FileExistsError, "Quantized output already exists"):
            self.preflight()
        self.assertEqual(self.qdq.read_bytes(), b"frozen quantized model")

    def test_fresh_export_never_overwrites_graphs_or_sidecars(self):
        for path in (self.fp32, self.fp32.with_suffix(".json"), self.qdq.with_suffix(".json")):
            with self.subTest(path=path.name):
                path.write_bytes(b"preserve")
                with self.assertRaises(FileExistsError):
                    self.preflight()
                self.assertEqual(path.read_bytes(), b"preserve")
                path.unlink()

    def test_new_bucket_can_extend_compatible_manifest_but_cannot_replace_existing_bucket(self):
        manifest = self.preflight()
        manifest["buckets"] = {"768": "frozen_768.onnx"}
        manifest["model_sha256"] = {"768": "a" * 64}
        path = self.output / "manifest.json"
        path.write_text(json.dumps(manifest))
        self.assertEqual(self.preflight()["buckets"], manifest["buckets"])
        manifest["buckets"]["1024"] = "other_frozen_name.onnx"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(FileExistsError, "already contains bucket 1024"):
            self.preflight()

    def test_main_rejects_conflicting_manifest_before_model_load_or_output_write(self):
        manifest = self.preflight()
        manifest["mask_penalty"] = -50
        path = self.output / "manifest.json"
        path.write_text(json.dumps(manifest))
        self.qdq.write_bytes(b"existing model must survive policy failure")
        before = {item.name: item.read_bytes() for item in self.output.iterdir()}
        load = Mock(side_effect=AssertionError("Preflight must precede model loading"))
        argv = ["build_fidelity.py", "--model", str(self.model), "--parquet", str(self.dataset),
                "--output-dir", str(self.output), "--length", "1024", "--zero-pad-embeddings",
                "--refine-matmul-rhs", "all"]
        with patch.object(sys, "argv", argv), patch.object(self.builder.laya, "load", load, create=True), \
                patch.object(self.builder.torch, "set_num_threads", Mock(), create=True), \
                patch.object(self.builder.torch, "set_num_interop_threads", Mock(), create=True):
            with self.assertRaisesRegex(ValueError, "different mask_penalty policy"):
                self.builder.main()
        load.assert_not_called()
        self.assertEqual(before, {item.name: item.read_bytes() for item in self.output.iterdir()})

    def test_corrected_manifest_cannot_be_extended_by_uncorrected_build(self):
        manifest = self.preflight()
        manifest["htp_conv_offset_correction"] = {"768": {}}
        (self.output / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "corrected model manifest"):
            self.preflight()


if __name__ == "__main__":
    unittest.main()
