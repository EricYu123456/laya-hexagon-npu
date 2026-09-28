"""Persistent EPContext configuration/publication tests with a fake QNN runtime."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from laya_npu import QNNEncoder
from npu.fidelity_runtime import qnn_backend_fingerprint


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(local_attention=4, pad_token_id=0, hidden_size=3)
        self.embeddings = torch.nn.Embedding(9, 3)

    def get_input_embeddings(self):
        return self.embeddings


class FakeOptions:
    def __init__(self):
        self.config = {}
        self.providers = None

    def add_session_config_entry(self, key, value):
        self.config[key] = value

    def add_provider_for_devices(self, devices, options):
        self.providers = (devices, dict(options))


class FakeSession:
    def __init__(self, bucket, wrong_shape=False):
        self.bucket = bucket
        self.wrong_shape = wrong_shape
        self.fallback_disabled = False

    def disable_fallback(self):
        self.fallback_disabled = True

    def get_inputs(self):
        bucket = self.bucket + int(self.wrong_shape)
        return [
            SimpleNamespace(name="inputs_embeds", shape=[1, bucket, 3], type="tensor(float)"),
            SimpleNamespace(name="attn_mask", shape=[1, 1, bucket, bucket], type="tensor(float)"),
            SimpleNamespace(name="sliding_mask", shape=[1, 1, bucket, bucket], type="tensor(float)"),
        ]


class FakeOrt:
    __version__ = "1.30.0"
    SessionOptions = FakeOptions

    def __init__(self):
        self.calls = []
        self.sessions = []
        self.fail_after_write = False
        self.omit_cache_write = False
        self.wrong_shape = False

    @staticmethod
    def get_ep_devices():
        return [SimpleNamespace(ep_name="QNNExecutionProvider")]

    def InferenceSession(self, path, sess_options):
        path = Path(path)
        self.calls.append((path, sess_options))
        bucket = json.loads(path.read_text())["bucket"]
        if sess_options.config.get("ep.context_enable") == "1":
            destination = Path(sess_options.config["ep.context_file_path"])
            if not self.omit_cache_write:
                destination.write_text(json.dumps({"bucket": bucket}))
            if self.fail_after_write:
                raise RuntimeError("simulated compile failure after a partial context write")
        session = FakeSession(bucket, self.wrong_shape)
        self.sessions.append(session)
        return session


class QNNContextCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="laya-cache-test-")
        self.root = Path(self.temporary.name)
        self.cache_dir = self.root / "cache"
        self.models = {}
        for bucket in (8, 16):
            path = self.root / f"source_{bucket}.onnx"
            path.write_text(json.dumps({"bucket": bucket}))
            self.models[bucket] = path
        self.ort = FakeOrt()
        self.backend = self.root / "libQnnHtp.so"
        self.backend.write_bytes(b"fake HTP backend")
        self.qnn = SimpleNamespace(__version__="2.5.0", get_qnn_htp_path=lambda: str(self.backend))
        self.runtime = patch("laya_npu._qnn_runtime", return_value=(self.ort, self.qnn))
        self.runtime.start()

    def tearDown(self):
        self.runtime.stop()
        self.temporary.cleanup()

    def encoder(self, **kwargs):
        return QNNEncoder(TinyEncoder(), self.models, cache_dir=self.cache_dir, **kwargs)

    def test_generation_is_atomic_and_evicted_bucket_loads_context(self):
        encoder = self.encoder()
        first = encoder._get_session(8)
        self.assertIs(first, encoder._get_session(8))
        encoder._get_session(16)
        loaded = encoder._get_session(8)
        self.assertIsNot(loaded, first)
        self.assertEqual(encoder.stats["context_cache_misses"], 2)
        self.assertEqual(encoder.stats["context_cache_writes"], 2)
        self.assertEqual(encoder.stats["context_cache_hits"], 1)
        self.assertEqual(encoder.stats["session_evictions"], 2)
        self.assertEqual(encoder.stats["cpu_fallbacks"], 0)
        self.assertEqual(len(list(self.cache_dir.iterdir())), 2)
        self.assertEqual(self.ort.calls[0][0], self.models[8])
        generated_options = self.ort.calls[0][1]
        self.assertEqual(generated_options.config["ep.context_embed_mode"], "1")
        self.assertFalse(Path(generated_options.config["ep.context_file_path"]).exists())
        cache_path, loaded_options = self.ort.calls[-1]
        self.assertEqual(cache_path, Path(encoder.stats["context_cache_paths"]["8"]))
        self.assertEqual(loaded_options.config["ep.context_enable"], "0")
        self.assertEqual(loaded_options.config["ep.context_file_path"], str(cache_path))
        self.assertTrue(cache_path.is_file())
        if os.name != "nt":
            self.assertEqual(cache_path.stat().st_mode & 0o777, 0o600)
        for _, options in self.ort.calls:
            self.assertEqual(options.config["session.disable_cpu_ep_fallback"], "1")
            self.assertEqual(options.providers[1]["htp_arch"], "68")
        self.assertTrue(all(session.fallback_disabled for session in self.ort.sessions))

    def test_cache_survives_new_instance_and_invalidates_on_graph_and_runtime_versions(self):
        original = self.encoder()
        original._get_session(8)
        original_path = original.stats["context_cache_paths"]["8"]
        fresh = self.encoder()
        fresh._get_session(8)
        self.assertEqual(fresh.stats["context_cache_hits"], 1)

        for changed in ("graph", "qnn", "ort"):
            with self.subTest(changed=changed):
                if changed == "graph":
                    self.models[8].write_text(json.dumps({"bucket": 8, "revision": 2}))
                elif changed == "qnn":
                    self.qnn.__version__ = "2.6.0"
                else:
                    self.ort.__version__ = "1.31.0"
                updated = self.encoder()
                updated._get_session(8)
                self.assertEqual(updated.stats["context_cache_misses"], 1)
                self.assertNotEqual(updated.stats["context_cache_paths"]["8"], original_path)
                original_path = updated.stats["context_cache_paths"]["8"]
        options = {"backend_path": str(self.backend), "htp_arch": "68"}
        key68 = fresh._context_cache_path(8, self.ort, self.qnn, options)
        key73 = fresh._context_cache_path(8, self.ort, self.qnn, {**options, "htp_arch": "73"})
        self.assertNotEqual(key68, key73)

    def test_precomputed_digest_reuses_verified_artifact_hash(self):
        digest = hashlib.sha256(self.models[8].read_bytes()).hexdigest()
        encoder = self.encoder(model_hashes={8: digest})
        with patch("laya_npu.sha256_file", side_effect=AssertionError("must not rehash")):
            encoder._get_session(8)
        self.assertEqual(encoder.stats["context_cache_writes"], 1)

    def test_disabled_cache_uses_source_without_hashing_or_creating_directory(self):
        with patch.dict(os.environ, {"LAYA_NPU_CONTEXT_CACHE": "0"}):
            encoder = self.encoder()
        with patch("laya_npu.sha256_file", side_effect=AssertionError("must not hash disabled cache")):
            session = encoder._get_session(8)
        self.assertTrue(session.fallback_disabled)
        self.assertEqual(self.ort.calls[0][0], self.models[8])
        self.assertNotIn("ep.context_enable", self.ort.calls[0][1].config)
        self.assertFalse(self.cache_dir.exists())
        self.assertFalse(encoder.stats["context_cache_enabled"])

    def test_corrupt_cache_is_actionable_and_never_retries_on_cpu_or_source(self):
        encoder = self.encoder()
        encoder._get_session(8)
        cache_path = Path(encoder.stats["context_cache_paths"]["8"])
        cache_path.write_text("corrupt context")
        calls_before = len(self.ort.calls)
        fresh = self.encoder()
        with self.assertRaisesRegex(RuntimeError, "Remove this exact cache file"):
            fresh._get_session(8)
        self.assertEqual(len(self.ort.calls), calls_before + 1)
        self.assertEqual(self.ort.calls[-1][0], cache_path)
        self.assertIsNone(fresh._session)
        self.assertEqual(fresh.stats["context_cache_errors"], 1)
        self.assertEqual(fresh.stats["cpu_fallbacks"], 0)

    def test_failed_partial_or_missing_generation_never_publishes_cache(self):
        for failure in ("partial", "missing", "bad_shape"):
            with self.subTest(failure=failure):
                self.ort.fail_after_write = failure == "partial"
                self.ort.omit_cache_write = failure == "missing"
                self.ort.wrong_shape = failure == "bad_shape"
                encoder = self.encoder()
                with self.assertRaisesRegex(RuntimeError, "QNN context generation failed"):
                    encoder._get_session(8)
                self.assertEqual(list(self.cache_dir.iterdir()), [])
                self.assertIsNone(encoder._session)
                self.assertEqual(encoder.stats["context_cache_writes"], 0)

    def test_cache_hit_still_checks_input_shapes(self):
        encoder = self.encoder()
        encoder._get_session(8)
        self.ort.wrong_shape = True
        fresh = self.encoder()
        with self.assertRaisesRegex(RuntimeError, "QNN context cache could not be loaded") as caught:
            fresh._get_session(8)
        self.assertIsInstance(caught.exception.__cause__, ValueError)
        self.assertEqual(fresh.stats["context_cache_hits"], 0)

    def correction(self):
        return {"8": {
            "output_sha256": hashlib.sha256(self.models[8].read_bytes()).hexdigest(),
            "runtime": qnn_backend_fingerprint("1.30.0", "2.5.0", self.backend),
        }}

    def test_corrected_graph_checks_fingerprint_with_cache_enabled_or_disabled(self):
        correction = self.correction()
        # Relocation is allowed: recorded file paths are informational, hashes bind bytes.
        correction["8"]["runtime"]["backend_path"] = "/previous/installation/libQnnHtp.so"
        for enabled in (True, False):
            with self.subTest(cache=enabled):
                encoder = self.encoder(cache_enabled=enabled, correction_metadata=correction)
                encoder._get_session(8)
                self.assertEqual(encoder.stats["correction_backend_verified_buckets"], [8])
                self.assertEqual(encoder.stats["backend_fingerprint"]["backend_sha256"],
                                 correction["8"]["runtime"]["backend_sha256"])
                json.dumps(encoder.stats)

    def test_wrong_backend_rejected_before_existing_cache_or_source_session(self):
        # Populate a valid cache first; provenance must not be skipped on a hit.
        self.encoder()._get_session(8)
        for field, wrong in (("onnxruntime", "0.0"), ("onnxruntime_qnn", "0.0"),
                             ("htp_arch", 73), ("backend_sha256", "0" * 64),
                             ("stub_sha256", "0" * 64), ("skel_sha256", "0" * 64)):
            for enabled in (True, False):
                with self.subTest(field=field, cache=enabled):
                    correction = self.correction()
                    correction["8"]["runtime"][field] = wrong
                    encoder = self.encoder(cache_enabled=enabled, correction_metadata=correction)
                    previous = len(self.ort.calls)
                    with self.assertRaisesRegex(RuntimeError, f"backend mismatch.*{field}"):
                        encoder._get_session(8)
                    self.assertEqual(len(self.ort.calls), previous)
                    self.assertEqual(encoder.stats["context_cache_hits"], 0)
                    self.assertEqual(encoder.stats["cpu_fallbacks"], 0)

    def test_same_versions_different_binary_invalidates_cache(self):
        previous = self.encoder()
        previous._get_session(8)
        previous_path = previous.stats["context_cache_paths"]["8"]
        self.backend.write_bytes(b"different SDK binary with same package versions")
        fresh = self.encoder()
        fresh._get_session(8)
        self.assertEqual(fresh.stats["context_cache_misses"], 1)
        self.assertNotEqual(fresh.stats["context_cache_paths"]["8"], previous_path)

    def test_corrected_graph_hash_is_checked_even_without_context_cache(self):
        correction = self.correction()
        self.models[8].write_text(json.dumps({"bucket": 8, "wrong_graph": True}))
        encoder = self.encoder(cache_enabled=False, correction_metadata=correction)
        with self.assertRaisesRegex(ValueError, "correction output_sha256 mismatch"):
            encoder._get_session(8)
        self.assertEqual(self.ort.calls, [])

    def test_missing_or_invalid_correction_runtime_rejected_at_construction(self):
        for invalid in ({}, {"8": {}}, {"8": {"runtime": {}}}, {"9": {}}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.encoder(correction_metadata=invalid)
        correction = self.correction()
        for field in ("backend_sha256", "onnxruntime", "onnxruntime_qnn", "htp_arch", "cpu_ep_fallback"):
            invalid = copy.deepcopy(correction)
            del invalid["8"]["runtime"][field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                self.encoder(correction_metadata=invalid)

    def test_auxiliary_binary_change_invalidates_cache_and_recorded_correction(self):
        for name in ("libQnnHtpV68Stub.so", "libQnnHtpV68Skel.so"):
            (self.root / name).write_bytes(b"original auxiliary")
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": str(self.root), "ADSP_LIBRARY_PATH": str(self.root)}):
            correction = self.correction()
            original = self.encoder(correction_metadata=correction)
            original._get_session(8)
            (self.root / "libQnnHtpV68Skel.so").write_bytes(b"changed skeleton")
            rejected = self.encoder(correction_metadata=correction)
            previous_calls = len(self.ort.calls)
            with self.assertRaisesRegex(RuntimeError, "skel_sha256"):
                rejected._get_session(8)
            self.assertEqual(len(self.ort.calls), previous_calls)
            uncorrected = self.encoder()
            uncorrected._get_session(8)
            self.assertNotEqual(uncorrected.stats["context_cache_paths"]["8"],
                                original.stats["context_cache_paths"]["8"])


if __name__ == "__main__":
    unittest.main()
