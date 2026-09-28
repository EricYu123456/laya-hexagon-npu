"""Backend correction provenance tests using files, never an NPU session."""

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from npu.fidelity_runtime import (
    artifact_metadata, correction_runtime_requirements, load_bucket_manifest,
    qnn_backend_fingerprint, qnn_backend_identity, validate_correction_backend,
)


class BackendFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="laya-fingerprint-")
        self.root = Path(self.temporary.name)
        self.backend = self.root / "libQnnHtp.so"
        self.backend.write_bytes(b"backend")
        self.environment = patch.dict(os.environ, {"LD_LIBRARY_PATH": "", "ADSP_LIBRARY_PATH": ""})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def fingerprint(self, **kwargs):
        return qnn_backend_fingerprint("1.30.0", "2.5.0", self.backend, **kwargs)

    def test_legacy_backend_only_fingerprint_is_enforced_without_inventing_history(self):
        actual = self.fingerprint()
        self.assertNotIn("skel_sha256", actual)
        expected = copy.deepcopy(actual)
        expected["backend_path"] = "/calibration/device/libQnnHtp.so"
        expected["platform"] = "historical OS description"
        validate_correction_backend(expected, actual, 768)
        self.assertEqual(qnn_backend_identity(expected), qnn_backend_identity(actual))
        expected["backend_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "backend_sha256"):
            validate_correction_backend(expected, actual, 768)
        json.dumps(actual, allow_nan=False)

    def test_configured_loader_order_and_required_auxiliary_binaries(self):
        with self.assertRaisesRegex(FileNotFoundError, "Stub"):
            self.fingerprint(require_auxiliary=True)
        first = self.root / "configured"
        first.mkdir()
        for label, filename in (("stub", "libQnnHtpV68Stub.so"), ("skel", "libQnnHtpV68Skel.so")):
            (self.root / filename).write_bytes(b"package fallback")
            (first / filename).write_bytes(label.encode())
        # Files beside the backend do not establish the loader's search path.
        with self.assertRaisesRegex(FileNotFoundError, "Stub"):
            self.fingerprint(require_auxiliary=True)
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": str(first), "ADSP_LIBRARY_PATH": str(first)}):
            actual = self.fingerprint(require_auxiliary=True)
        self.assertEqual(actual["stub_path"], str((first / "libQnnHtpV68Stub.so").resolve()))
        self.assertEqual(actual["skel_sha256"], hashlib.sha256(b"skel").hexdigest())
        self.assertEqual(actual["backend_sha256"], hashlib.sha256(b"backend").hexdigest())

    def test_manifest_graph_binding_is_required_even_without_model_sha256(self):
        graph = self.root / "encoder.onnx"
        graph.write_bytes(b"corrected graph")
        checkpoint = self.root / "model.safetensors"
        checkpoint.write_bytes(b"checkpoint")
        manifest = {"buckets": {"768": graph.name}, "htp_conv_offset_correction": {"768": {
            "runtime": self.fingerprint(), "output_sha256": hashlib.sha256(graph.read_bytes()).hexdigest(),
        }}}
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest))
        paths, loaded = load_bucket_manifest(path)
        artifact_metadata(path, loaded, paths, checkpoint)
        graph.write_bytes(b"different graph")
        with self.assertRaisesRegex(ValueError, "output_sha256 mismatch"):
            artifact_metadata(path, loaded, paths, checkpoint)

    def test_manifest_cannot_silently_disable_an_invalid_correction_record(self):
        for value in (None, {}, [], {"768": {"runtime": {}}}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                correction_runtime_requirements({"htp_conv_offset_correction": value}, {768: "model"})
        self.assertEqual(correction_runtime_requirements({}, {768: "model"}), {})


if __name__ == "__main__":
    unittest.main()
