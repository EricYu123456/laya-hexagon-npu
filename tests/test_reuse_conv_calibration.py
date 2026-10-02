"""Synthetic CPU contracts for exact-probe calibration reuse; no NPU used."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh

from npu import calibrate_conv_offsets as cal
from npu import reuse_conv_calibration as reuse_module


def model():
    arrays = {"xs": np.array(.01, np.float32), "xz": np.array(32768, np.uint16),
              "wq": np.ones((2, 3, 1, 1), np.int8), "ws": np.array([.01, .02], np.float32),
              "wz": np.zeros(2, np.int8), "ys": np.array(.001, np.float32), "yz": np.array(32768, np.uint16)}
    nodes = [h.make_node("QuantizeLinear", ["x", "xs", "xz"], ["xq"]),
             h.make_node("DequantizeLinear", ["xq", "xs", "xz"], ["xdq"]),
             h.make_node("DequantizeLinear", ["wq", "ws", "wz"], ["wdq"], axis=0),
             h.make_node("Conv", ["xdq", "wdq"], ["y"], name="conv", kernel_shape=[1, 1]),
             h.make_node("QuantizeLinear", ["y", "ys", "yz"], ["yq"]),
             h.make_node("DequantizeLinear", ["yq", "ys", "yz"], ["ydq"])]
    return h.make_model(h.make_graph(nodes, "reuse_test", [h.make_tensor_value_info("x", T.FLOAT, [1, 3, 5, 1])],
                                     [h.make_tensor_value_info("ydq", T.FLOAT, [1, 2, 5, 1])],
                                     [nh.from_array(value, name) for name, value in arrays.items()]),
                        opset_imports=[h.make_opsetid("", 21)], ir_version=10)


class CalibrationReuseTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.donor, self.out = self.root / "donor", self.root / "recipient"
        self.source, self.recipient_source = self.root / "source.onnx", self.root / "recipient.onnx"
        original = model()
        onnx.save(original, self.source)
        original.doc_string = "Different non-Conv graph metadata; same extracted probes"
        onnx.save(original, self.recipient_source)
        for source, out in ((self.source, self.donor), (self.recipient_source, self.out)):
            with contextlib.redirect_stdout(io.StringIO()):
                cal.prepare(argparse.Namespace(source=str(source), out=str(out), validation_count=1))
        self.manifest = json.loads((self.donor / "probe_manifest.json").read_text())
        self.runtime = {"onnxruntime": "synthetic-test", "onnxruntime_qnn": "synthetic-test", "htp_arch": 68,
                        "backend_sha256": "1" * 64, "stub_sha256": "2" * 64, "skel_sha256": "3" * 64,
                        "cpu_ep_fallback": False}
        patcher = mock.patch.object(reuse_module, "current_runtime_fingerprint", return_value=self.runtime)
        self.fingerprint = patcher.start()
        self.addCleanup(patcher.stop)
        delta = np.array([.01, -.02], np.float32)
        zero = np.zeros((1, 2, 1, 1), np.float32)
        self.arrays = {"cpu_conv_000_small_y": zero, "htp_conv_000_small_y": delta.reshape(1, 2, 1, 1),
                       "cpu_conv_000_original_y": np.zeros((1, 2, 5, 1), np.float32),
                       "htp_conv_000_original_y": np.broadcast_to(delta.reshape(1, 2, 1, 1), (1, 2, 5, 1)).copy()}
        np.savez(self.donor / "zero_outputs.npz", **self.arrays)
        np.savez(self.donor / "correction_biases.npz", conv_000=-delta)
        scale = self.manifest["records"][0]["output_scale"]
        self.result = {"source_sha256": self.manifest["source_sha256"], "probe_sha256": self.manifest["probe_sha256"],
                       "probe_manifest_sha256": cal.sha256(self.donor / "probe_manifest.json"),
                       "correction_biases_sha256": cal.sha256(self.donor / "correction_biases.npz"),
                       "runtime": self.runtime, "all_shape_validations_passed": True,
                       "per_conv": [{"id": "conv_000", "node": "conv", "mean_abs_offset": float(np.abs(delta).mean()),
                                     "max_abs_offset": float(np.abs(delta).max()), "output_step": scale}],
                       "shape_validation": [{"id": "conv_000", "passed": True, "max_abs_offset_difference": 0.,
                                             "tolerance": scale * .25 + 1e-7}]}
        self.write_result()

    def write_result(self):
        (self.donor / "calibration_results.json").write_text(json.dumps(self.result))

    def reuse(self):
        return reuse_module.reuse(from_dir=self.donor, out=self.out, source=self.recipient_source, from_source=self.source)

    def assert_no_measurements(self):
        for name in ("zero_outputs.npz", "correction_biases.npz", "calibration_results.json"):
            self.assertFalse((self.out / name).exists())

    def test_exact_probe_rebinds_evidence_and_applies_to_recipient(self):
        donor_result = (self.donor / "calibration_results.json").read_bytes()
        rebound = self.reuse()
        self.assertEqual(rebound["source_sha256"], cal.sha256(self.recipient_source))
        self.assertNotEqual(rebound["source_sha256"], rebound["calibration_reuse"]["donor_source_sha256"])
        self.assertEqual(rebound["probe_manifest_sha256"], cal.sha256(self.out / "probe_manifest.json"))
        self.assertEqual(rebound["calibration_reuse"]["donor_calibration_results_sha256"], cal.sha256(self.donor / "calibration_results.json"))
        self.assertEqual((self.donor / "calibration_results.json").read_bytes(), donor_result)
        self.assertEqual((self.out / "zero_outputs.npz").read_bytes(), (self.donor / "zero_outputs.npz").read_bytes())
        with contextlib.redirect_stdout(io.StringIO()):
            cal.apply(argparse.Namespace(source=str(self.recipient_source), out=str(self.out),
                                         target=str(self.root / "corrected.onnx"), source_manifest=None))
        self.assertTrue((self.root / "corrected.onnx").is_file())
        provenance = json.loads((self.root / "corrected.offsets.json").read_text())
        self.assertEqual(provenance["calibration_reuse"], rebound["calibration_reuse"])
        with self.assertRaises(FileExistsError):
            self.reuse()

    def test_probe_or_record_changes_are_rejected_before_writes(self):
        probe = self.out / "zero_probes.onnx"
        original = probe.read_bytes()
        probe.write_bytes(original + b"different bytes")
        with self.assertRaisesRegex(ValueError, "byte-identical"):
            self.reuse()
        self.assert_no_measurements()
        probe.write_bytes(original)
        path = self.out / "probe_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["records"][0]["input_shape"][2] = 6  # Different bucket/shape must not pass on small probes alone.
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "records differ"):
            self.reuse()
        self.assert_no_measurements()

    def test_corrected_hash_cannot_hide_wrong_measurement_or_summary(self):
        np.savez(self.donor / "correction_biases.npz", conv_000=np.zeros(2, np.float32))
        self.result["correction_biases_sha256"] = cal.sha256(self.donor / "correction_biases.npz")
        self.write_result()
        with self.assertRaisesRegex(ValueError, "negative measured offset"):
            self.reuse()
        self.assert_no_measurements()

    def test_recomputes_spatial_check_instead_of_trusting_flags(self):
        self.arrays["htp_conv_000_original_y"][0, 0, 0, 0] += .1
        np.savez(self.donor / "zero_outputs.npz", **self.arrays)
        with self.assertRaisesRegex(ValueError, "spatial checks disagree"):
            self.reuse()
        self.assert_no_measurements()

    def test_runtime_mismatch_missing_fingerprint_and_chained_reuse_rejected(self):
        self.fingerprint.return_value = {**self.runtime, "backend_sha256": "4" * 64}
        with self.assertRaisesRegex(ValueError, "runtime identity differs"):
            self.reuse()
        self.assert_no_measurements()
        self.fingerprint.return_value = self.runtime
        self.result["runtime"] = {"cpu_ep_fallback": False}
        self.write_result()
        with self.assertRaisesRegex(ValueError, "complete strict-HTP"):
            self.reuse()
        self.assert_no_measurements()
        self.result["runtime"] = self.runtime
        self.result["calibration_reuse"] = {}
        self.write_result()
        with self.assertRaisesRegex(ValueError, "original measured donor"):
            self.reuse()
        self.assert_no_measurements()


if __name__ == "__main__":
    unittest.main()
