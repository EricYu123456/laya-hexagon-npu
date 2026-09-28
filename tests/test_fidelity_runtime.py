"""Input fidelity checks that do not require PyTorch or an ARM QNN provider."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from npu.fidelity_runtime import artifact_metadata, attention_masks, load_bucket_manifest, select_bucket


class FidelityInputTests(unittest.TestCase):
    def test_bucket_boundary_never_truncates(self):
        buckets = {512: "b", 128: "a", 1024: "c"}
        self.assertEqual(select_bucket(buckets, 128), 128)
        self.assertEqual(select_bucket(buckets, 129), 512)
        self.assertEqual(select_bucket(buckets, 631), 1024)
        with self.assertRaisesRegex(ValueError, "1025 tokens"):
            select_bucket(buckets, 1025)

    def test_local_boundary_and_padding_union(self):
        global_mask, local_mask = attention_masks(128, 100, local_radius=64, mask_penalty=-100)
        self.assertEqual(global_mask.shape, (1, 1, 128, 128))
        self.assertEqual(global_mask.dtype, np.float32)
        self.assertEqual(global_mask[0, 0, 0, 99], 0)
        self.assertTrue(np.all(global_mask[..., 100:] == -100))
        self.assertEqual(local_mask[0, 0, 0, 64], 0)
        self.assertEqual(local_mask[0, 0, 0, 65], -100)
        self.assertEqual(local_mask[0, 0, 99, 35], 0)
        self.assertEqual(local_mask[0, 0, 99, 34], -100)
        self.assertEqual(local_mask[0, 0, 0, 127], -100)
        self.assertTrue(np.all(local_mask[..., 100:] == -100))

    def test_short_bucket_local_equals_global(self):
        global_mask, local_mask = attention_masks(64, 47)
        np.testing.assert_array_equal(global_mask, local_mask)

    def test_invalid_mask_parameters_rejected(self):
        for penalty in [0, 1, float("nan"), float("-inf")]:
            with self.assertRaises(ValueError):
                attention_masks(64, 32, mask_penalty=penalty)
        with self.assertRaises(ValueError):
            attention_masks(64, 65)

    def test_manifest_paths_resolve_from_manifest_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "encoder.onnx").touch()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"buckets": {"512": "encoder.onnx"}, "mask_penalty": -50}))
            paths, metadata = load_bucket_manifest(manifest)
            self.assertEqual(paths, {512: (root / "encoder.onnx").resolve()})
            self.assertEqual(metadata["mask_penalty"], -50)
            manifest.write_text(json.dumps({"buckets": {"512": "missing.onnx"}}))
            with self.assertRaisesRegex(FileNotFoundError, "bucket 512"):
                load_bucket_manifest(manifest)

    def test_manifest_binds_graph_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = root / "encoder.onnx"
            graph.write_bytes(b"exported graph")
            checkpoint = root / "model.safetensors"
            checkpoint.write_bytes(b"original checkpoint")
            manifest = {
                "buckets": {"512": graph.name},
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "model_sha256": {"512": hashlib.sha256(graph.read_bytes()).hexdigest()},
                "precision": "a16w8", "source_checkpoint": "multilingual",
            }
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            provenance = artifact_metadata(path, manifest, {512: graph}, checkpoint)
            self.assertTrue(provenance["checkpoint_manifest_verified"])
            self.assertTrue(provenance["models"]["512"]["manifest_verified"])
            self.assertEqual(provenance["precision"], "a16w8")
            self.assertEqual(provenance["supported_buckets"], [512])

            # Formatting changes affect the file hash, while the canonical hash
            # remains stable so provenance can distinguish content from layout.
            path.write_text(json.dumps(manifest, indent=2))
            reformatted = artifact_metadata(path, manifest, {512: graph}, checkpoint)
            self.assertNotEqual(provenance["manifest_sha256"], reformatted["manifest_sha256"])
            self.assertEqual(provenance["manifest_canonical_sha256"], reformatted["manifest_canonical_sha256"])

            graph.write_bytes(b"different graph")
            with self.assertRaisesRegex(ValueError, r"model_sha256\[512\] mismatch"):
                artifact_metadata(path, manifest, {512: graph}, checkpoint)
            checkpoint.write_bytes(b"wrong language checkpoint")
            with self.assertRaisesRegex(ValueError, "checkpoint_sha256 mismatch"):
                artifact_metadata(path, manifest, {512: graph}, checkpoint)


if __name__ == "__main__":
    unittest.main()
