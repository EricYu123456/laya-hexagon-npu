import json
import tempfile
import unittest
from pathlib import Path

from npu.fidelity_runtime import load_bucket_manifest, sha256_file
from npu.merge_manifests import merge_manifests


class MergeManifestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)

    def source(self, bucket, *, zero_pad=True, corrected=False):
        directory = self.root / str(bucket)
        directory.mkdir(exist_ok=True)
        graph = directory / "graph.onnx"
        graph.write_bytes(str(bucket).encode())
        value = {"checkpoint_sha256": "a" * 64, "mask_penalty": -100.0,
                 "zero_pad_embeddings": zero_pad, "precision": "a16w8",
                 "buckets": {str(bucket): graph.name},
                 "model_sha256": {str(bucket): sha256_file(graph)}}
        if corrected:
            value["htp_conv_offset_correction"] = {str(bucket): {
                "output_sha256": sha256_file(graph),
                "runtime": {"onnxruntime": "1.30.0", "onnxruntime_qnn": "2.5.0",
                            "htp_arch": 68, "cpu_ep_fallback": False,
                            "backend_sha256": "c" * 64},
            }}
        path = directory / "manifest.json"
        path.write_text(json.dumps(value))
        return path

    def test_merge_resolves_paths_and_preserves_corrections(self):
        first, second = self.source(768, corrected=True), self.source(1024, corrected=True)
        target = self.root / "deploy/manifest.json"
        merged = merge_manifests([first, second], target)
        paths, _ = load_bucket_manifest(target)
        self.assertEqual(set(paths), {768, 1024})
        self.assertEqual(set(merged["htp_conv_offset_correction"]), {"768", "1024"})
        self.assertEqual(paths[768], self.root / "768/graph.onnx")

    def test_rejects_policy_and_bucket_conflicts(self):
        first, second = self.source(768), self.source(1024, zero_pad=False)
        with self.assertRaisesRegex(ValueError, "policies"):
            merge_manifests([first, second], self.root / "bad.json")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            merge_manifests([first, first], self.root / "bad.json")
        self.assertFalse((self.root / "bad.json").exists())

    def test_rejects_changed_model_and_correction(self):
        source = self.source(768, corrected=True)
        value = json.loads(source.read_text())
        value["htp_conv_offset_correction"]["768"]["output_sha256"] = "b" * 64
        source.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "provenance"):
            merge_manifests([source], self.root / "bad.json")
        (source.parent / "graph.onnx").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "model SHA256"):
            merge_manifests([source], self.root / "bad.json")

    def test_rejects_incompatible_corrected_backends(self):
        first, second = self.source(768, corrected=True), self.source(1024, corrected=True)
        value = json.loads(second.read_text())
        value["htp_conv_offset_correction"]["1024"]["runtime"]["backend_sha256"] = "d" * 64
        second.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "different backend"):
            merge_manifests([first, second], self.root / "bad.json")


if __name__ == "__main__":
    unittest.main()
