"""Regression checks for fidelity metrics; no model or accelerator is required."""
import argparse
import copy
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmark_fidelity import decision_metrics, main, model_fingerprint, read_jsonl, select_indices, summarize_decisions


def choice(probabilities, label=None):
    return {"type": "choice", "choice": label or max(probabilities, key=probabilities.get), "probabilities": probabilities, "confidence": 0.5, "action": {"act_probability": 0.9}}


class FidelityMetricTests(unittest.TestCase):
    def test_close_gold_accuracy_cannot_hide_completely_different_decisions(self):
        question = {"type": "choice", "criteria": {"a": "A", "b": "B"}}
        metric = decision_metrics(question, choice({"a": 0.9, "b": 0.1}), choice({"a": 0.1, "b": 0.9}))
        self.assertEqual(metric["decision_mismatch"], 1)
        self.assertAlmostEqual(metric["total_variation"], 0.8)
        self.assertEqual(summarize_decisions([metric])["decision_error_percent"], 100)

    def test_same_label_does_not_hide_probability_error(self):
        question = {"type": "choice", "criteria": ["a", "b", "c"]}
        metric = decision_metrics(question, choice({"a": 1.0, "b": 0.0, "c": 0.0}), choice({"c": 0.25, "b": 0.25, "a": 0.5}))
        self.assertEqual(metric["decision_mismatch"], 0)
        self.assertAlmostEqual(metric["total_variation"], 0.5)
        self.assertAlmostEqual(metric["probability_mae"], 1 / 3)

    def test_rounded_probability_ties_preserve_published_choice(self):
        question = {"type": "choice", "criteria": ["a", "b"]}
        metric = decision_metrics(question, choice({"a": 0.5, "b": 0.5}, "a"), choice({"a": 0.5, "b": 0.5}, "b"))
        self.assertEqual(metric["decision_mismatch"], 1)
        self.assertEqual(metric["argmax_mismatch"], 0)

    def test_noul_threshold_and_zero_probability(self):
        question = {"type": "noul"}
        ref = {"type": "noul", "noul": 0, "confidence": 1, "action": {"act_probability": 1}}
        cand = {"type": "noul", "noul": 0.51, "confidence": 0.51, "action": {"act_probability": 0.2}}
        metric = decision_metrics(question, ref, cand)
        self.assertEqual(metric["decision_mismatch"], 1)
        self.assertAlmostEqual(metric["total_variation"], 0.51)
        self.assertAlmostEqual(metric["action_probability_abs_error"], 0.8)

    def test_score_expectation_is_normalized_and_argmax_separate(self):
        question = {"type": "score", "criteria": ["low", "medium", "high"]}
        ref = {"type": "score", "score": 0.8, "probabilities": {"0": 0.5, "1": 0.2, "2": 0.3}, "confidence": 0.2, "action": {"act_probability": 1}}
        cand = copy.deepcopy(ref)
        cand.update(score=1.2, probabilities={"0": 0.3, "1": 0.2, "2": 0.5})
        metric = decision_metrics(question, ref, cand)
        self.assertEqual(metric["decision_mismatch"], 0)
        self.assertEqual(metric["argmax_mismatch"], 1)
        self.assertAlmostEqual(metric["score_normalized_abs_error"], 0.2)

    def test_invalid_or_missing_probability_is_rejected(self):
        question = {"type": "choice", "criteria": ["a", "b"]}
        reference = choice({"a": 0.5, "b": 0.5})
        for probabilities in ({"a": 0.5}, {"a": 0.7, "b": 0.7}, {"a": math.nan, "b": 0.5}, {"a": -0.1, "b": 1.1}):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                decision_metrics(question, reference, choice(probabilities))

    def test_api_rounding_sum_is_normalized(self):
        question = {"type": "choice", "criteria": ["a", "b", "c"]}
        answer = choice({"a": 0.3333, "b": 0.3333, "c": 0.3333})
        self.assertEqual(decision_metrics(question, answer, answer)["total_variation"], 0)

    def test_indices_cannot_silently_duplicate_or_omit_bad_cases(self):
        args = argparse.Namespace(indices="0,100,200,300", offset=0, limit=None)
        self.assertEqual(select_indices(400, args), [0, 100, 200, 300])
        args.indices = "0,0"
        with self.assertRaises(ValueError):
            select_indices(400, args)
        args.indices = None
        args.limit = -1
        with self.assertRaises(ValueError):
            select_indices(400, args)

    def test_predeclared_exclusion_selection(self):
        args = argparse.Namespace(indices=None, offset=0, limit=None, exclude_indices="0,2")
        self.assertEqual(select_indices(4, args), [1, 3])
        args.exclude_indices = "0,1,2,3"
        with self.assertRaises(ValueError):
            select_indices(4, args)

    def test_semantic_model_config_hash_survives_newlines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "encoder").mkdir()
            (root / "tokenizer").mkdir()
            (root / "model.safetensors").write_bytes(b"weights")
            names = ["rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]
            for name in names:
                (root / name).write_bytes(b'{\n  "a": 1, "b": 2\n}\n')
            first = model_fingerprint(root)
            for name in names:
                (root / name).write_bytes(b'{\r\n  "b": 2, "a": 1\r\n}\r\n')
            self.assertEqual(first, model_fingerprint(root))

    def test_reference_capture_resume_and_comparison_pipeline(self):
        answer = {"answers": {"q": choice({"a": 0.8, "b": 0.2})}, "usage": {"input_tokens": 2}}
        fake_agent = SimpleNamespace(predict=lambda state, questions: answer)
        fake_module = SimpleNamespace(__file__=__file__)
        rows = [{"id": f"case{i}", "workflow": "test", "state": json.dumps("state"), "questions": json.dumps({"q": {"type": "choice", "criteria": ["a", "b"], "instructions": "pick"}})} for i in range(2)]
        tokens = {"q": {"sha256": "sequence", "tokens": 2, "markers": [0, 1]}}
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            for name, value in (("load_dataset", rows), ("model_fingerprint", {"model": "hash"}), ("runtime_info", {"packages": {}, "laya_source_files": {}}), ("load_agent", (fake_agent, fake_module)), ("token_metadata", tokens), ("sha256_file", "hash")):
                stack.enter_context(patch("benchmark_fidelity." + name, return_value=value))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            reference = Path(directory) / "reference.jsonl"
            output = Path(directory) / "report.json"
            self.assertEqual(main(["--backend", "cpu", "--output", str(reference), "--limit", "1"]), 0)
            self.assertEqual(main(["--backend", "cpu", "--output", str(reference), "--resume"]), 0)
            _, cached = read_jsonl(reference)
            self.assertEqual(len(cached), 2)
            self.assertEqual(main(["--reference", str(reference), "--output", str(output)]), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(report["full_public_split_pass"])
            self.assertEqual(report["token_sequences_equal_cases"], 2)
            self.assertEqual(report["overall"]["total_variation"]["mean"], 0)
            self.assertTrue(output.with_suffix(".cases.jsonl").is_file())
            heldout = Path(directory) / "heldout.json"
            self.assertEqual(main(["--reference", str(reference), "--output", str(heldout), "--exclude-indices", "0"]), 0)
            report = json.loads(heldout.read_text(encoding="utf-8"))
            self.assertTrue(report["heldout_selection_pass"])
            self.assertFalse(report["full_public_split_pass"])
            self.assertEqual(report["evaluation_scope"], "heldout_selection")
            self.assertFalse(report["latency_same_host_and_threads"])


if __name__ == "__main__":
    unittest.main()
