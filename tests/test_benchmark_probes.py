import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark_probes as probes


class ProbeTests(unittest.TestCase):
    def test_mixed_lengths_share_collated_batch_bucket(self):
        self.assertEqual(probes.expected_bucket_calls({"a": {"tokens": 760}, "b": {"tokens": 790}},
                                                      [768, 1024]), {"1024": 2})

    def test_stale_npu_counters_cannot_certify_a_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = root / "suite.json"
            suite.write_text(json.dumps([{"id": "one", "state": "x", "questions": {
                "q": {"type": "noul", "instructions": "Refund?"}}}]))
            reference = root / "reference.json"
            answer = {"answers": {"q": {"type": "noul", "noul": .2, "confidence": .8,
                                         "action": {"act_probability": 1.0}}},
                      "usage": {"input_tokens": 40, "output_tokens": 0}}
            tokens = {"q": {"sha256": "unchanged", "tokens": 40, "markers": [1]}}

            class Agent:
                def __init__(self, count_calls=False):
                    self.count_calls = count_calls
                    self.npu_stats = {"npu_calls": 100, "cpu_fallbacks": 0, "bucket_calls": {"768": 100}}

                def predict(self, *_):
                    if self.count_calls:
                        self.npu_stats["npu_calls"] += 1
                        self.npu_stats["bucket_calls"]["768"] += 1
                    return answer

            with contextlib.ExitStack() as stack:
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(patch.object(probes, "model_fingerprint", return_value={"checkpoint": "same"}))
                stack.enter_context(patch.object(probes, "runtime_info", return_value={}))
                stack.enter_context(patch.object(probes, "token_metadata", return_value=tokens))
                stack.enter_context(patch.object(probes, "candidate_assets", return_value={"supported_buckets": [768, 1024]}))
                loader = stack.enter_context(patch.object(probes, "load_agent"))
                loader.return_value = Agent(), SimpleNamespace(__name__="laya")
                self.assertEqual(probes.main(["--suite", str(suite), "--backend", "cpu", "--output", str(reference)]), 0)
                loader.return_value = Agent(), SimpleNamespace(__name__="laya_npu")
                result = root / "stale.json"
                args = ["--suite", str(suite), "--reference", str(reference), "--output", str(result)]
                self.assertEqual(probes.main(args), 1)
                self.assertFalse(json.loads(result.read_text())["records"][0]["npu_execution_verified"])
                loader.return_value = Agent(count_calls=True), SimpleNamespace(__name__="laya_npu")
                args[-1] = str(root / "actual.json")
                self.assertEqual(probes.main(args), 0)

    def test_suite_requires_unique_ids_and_valid_length(self):
        probe = {"id": "one", "state": "x", "questions": {"q": {}}}
        with self.assertRaisesRegex(ValueError, "unique"):
            probes.validate_suite([probe, probe])
        with self.assertRaisesRegex(ValueError, "length"):
            probes.validate_suite([{**probe, "min_sequence_tokens": 1025}])


if __name__ == "__main__":
    unittest.main()
