"""Regression checks for fidelity metrics; no model or accelerator is required."""
import argparse
import copy
import contextlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchmark_fidelity import decision_metrics, main, model_fingerprint, read_jsonl, select_indices, summarize_decisions
import benchmark_fidelity as benchmark


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


class OfflineReportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.rows = [{"id": f"row{i}", "workflow": "test", "state": {"value": f"state{i}"},
                      "questions": {"q": {"type": "choice", "criteria": ["a", "b"]}}} for i in range(4)]
        self.dataset = {"id": benchmark.DATASET, "revision": benchmark.REVISION, "file": benchmark.PARQUET, "sha256": benchmark.PARQUET_SHA256}
        self.reference_metadata = {"kind": "metadata", "format_version": 1, "backend": "cpu", "module": "laya",
                                   "dataset": self.dataset, "model_files": {"weights": "unchanged"}, "threads": 4,
                                   "runtime": {"hostname": "wsl", "platform": "x86"}, "created_utc": "2025-01-01T00:00:00+00:00"}
        self.source_metadata = {**self.reference_metadata, "backend": "npu", "module": "laya_npu",
                                "runtime": {"hostname": "pi", "platform": "arm"}, "selection": [0, 1, 2, 3],
                                "total_dataset_cases": 4, "excluded_calibration_or_development_indices": [],
                                "candidate_assets": {"graph_sha256": "frozen-graph"}, "warmup_cases": 1,
                                "accelerator_stats_after_warmup": {"npu_calls": 1}}
        self.references, self.comparisons = [], []
        tokens = {"q": {"sha256": "identical-input", "tokens": 2, "markers": [0, 1]}}
        for index, row in enumerate(self.rows):
            identity = benchmark.parse_row(row, index)[0]
            cpu = {"answers": {"q": choice({"a": 0.9, "b": 0.1})}, "usage": {"input_tokens": 2}}
            candidate = copy.deepcopy(cpu)
            if index == 0:
                candidate["answers"]["q"] = choice({"a": 0.1, "b": 0.9})
            reference = {"kind": "reference", **identity, "cpu": cpu, "cpu_ms": 20 + index, "tokens": tokens}
            comparison = {"kind": "comparison", **identity, "cpu": cpu, "cpu_ms": 20 + index, "candidate": candidate,
                          "candidate_ms": 10 + index, "reference_tokens": tokens, "candidate_tokens": copy.deepcopy(tokens)}
            self.references.append(reference)
            self.comparisons.append(benchmark.recompute_comparison(comparison, reference, row, index))
        self.reference = self.root / "reference.jsonl"
        self.records = self.root / "full.cases.jsonl"
        self.source_report = self.root / "full.json"
        self.plan_path = self.root / "plan.json"
        self.write_records(self.reference, self.reference_metadata, self.references)
        self.write_records(self.records, self.source_metadata, self.comparisons)
        self.full_report = benchmark.aggregate_report(self.source_metadata, self.reference_metadata, benchmark.sha256_file(self.reference), 4,
                                                     self.comparisons, self.records, {"npu_calls": 5, "cpu_fallbacks": 0})
        self.source_report.write_text(json.dumps(self.full_report), encoding="utf-8")
        self.make_plan()

    @staticmethod
    def write_records(path, metadata, records):
        with path.open("w", encoding="utf-8") as stream:
            for item in (metadata, *records):
                benchmark.dump_line(stream, item)

    def make_plan(self):
        def git(*args, **kwargs):
            return subprocess.run(["git", "-C", str(self.root), *args], check=True, capture_output=True, **kwargs).stdout
        git("init", "-q")
        contents = b"The exclusions below were fixed before evaluation.\nEXCLUDED=0,2\n"
        (self.root / "method.md").write_bytes(contents)
        git("add", "method.md")
        env = {**os.environ, "GIT_AUTHOR_DATE": "2020-01-01T00:00:00+00:00", "GIT_COMMITTER_DATE": "2020-01-01T00:00:00+00:00"}
        git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false", "commit", "-qm", "Declare selection", env=env)
        # Read committed bytes so checkout newline conversion cannot affect the hash.
        contents = git("show", "HEAD:method.md")
        self.plan = {"kind": "heldout_selection_plan", "format_version": 1, "created_utc": "2026-01-01T00:00:00+00:00",
                     "dataset": self.dataset, "total_dataset_cases": 4, "selection": [1, 3],
                     "excluded_calibration_or_development_indices": [0, 2],
                     "declaration": {"kind": "git_documentation", "commit": git("rev-parse", "HEAD").decode().strip(),
                                     "path": "method.md", "document_sha256": benchmark.hashlib.sha256(contents).hexdigest(),
                                     "committed_utc": git("show", "-s", "--format=%cI", "HEAD").decode().strip()}}
        self.save_plan()

    def save_plan(self):
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")

    def derive(self, *, plan=True, source_report=True, extra=()):
        output = self.root / "heldout.json"
        args = ["--from-candidate-records", str(self.records), "--reference", str(self.reference), "--output", str(output)]
        if plan:
            args.extend(["--selection-plan", str(self.plan_path)])
        if source_report:
            args.extend(["--source-report", str(self.source_report)])
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(benchmark, "load_dataset", return_value=self.rows))
            original_verify = benchmark.verify_selection_plan
            stack.enter_context(patch.object(benchmark, "verify_selection_plan", side_effect=lambda path, metadata, total: original_verify(path, metadata, total, repository=self.root)))
            for name in ("load_agent", "model_fingerprint", "runtime_info"):
                stack.enter_context(patch.object(benchmark, name, side_effect=AssertionError("Offline derivation must not load a model/runtime")))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            code = main([*args, *extra])
        return code, json.loads(output.read_text(encoding="utf-8"))

    def test_historical_declaration_recomputes_subset_without_reassigning_run_counters(self):
        # Cached metrics/flags are untrusted, even when the original report was correct.
        self.comparisons[1]["metrics"]["q"]["decision_mismatch"] = 1
        self.comparisons[1]["token_sequences_equal"] = False
        self.write_records(self.records, self.source_metadata, self.comparisons)
        code, report = self.derive()
        self.assertEqual(code, 0)
        self.assertTrue(report["heldout_selection_pass"])
        self.assertFalse(report["full_public_split_pass"])
        self.assertEqual(report["overall"]["decisions"], 2)
        self.assertEqual(report["overall"]["decision_mismatches"], 0)
        self.assertEqual(report["metadata"]["selection"], [1, 3])
        self.assertEqual(report["metadata"]["candidate_assets"], self.source_metadata["candidate_assets"])
        self.assertEqual(report["original_run"]["metadata"], self.source_metadata)
        self.assertEqual(report["original_run"]["accelerator_stats_after_measurement"]["npu_calls"], 5)
        self.assertIsNone(report["accelerator_stats_after_measurement"])
        self.assertIsNone(report["metadata"]["accelerator_stats_after_warmup"])
        self.assertFalse(report["latency_same_host_and_threads"])
        self.assertEqual(report["latency_case_ms"]["candidate"]["mean"], 12)
        self.assertTrue(report["derivation"]["selection_plan"]["declaration_predates_source_run"])

    def test_posthoc_exclusions_do_not_certify_heldout(self):
        code, report = self.derive(plan=False, extra=("--exclude-indices", "0,2"))
        self.assertEqual(code, 0)
        self.assertTrue(report["selected_cases_pass"])
        self.assertFalse(report["heldout_selection_pass"])
        self.assertFalse(report["complete_declared_heldout_selection"])
        self.assertEqual(report["evaluation_scope"], "posthoc_subset")

    def test_incomplete_source_rejected_even_if_subset_is_present(self):
        self.write_records(self.records, self.source_metadata, self.comparisons[1:])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.derive()

    def test_duplicate_source_row_rejected(self):
        self.write_records(self.records, self.source_metadata, [*self.comparisons, self.comparisons[0]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.derive()

    def test_changed_reference_even_in_excluded_case_is_rejected(self):
        self.comparisons[0]["cpu"] = copy.deepcopy(self.comparisons[0]["candidate"])
        self.write_records(self.records, self.source_metadata, self.comparisons)
        with self.assertRaisesRegex(ValueError, "supplied reference"):
            self.derive()

    def test_candidate_token_mismatch_is_recomputed_and_fails_gate(self):
        self.comparisons[1]["candidate_tokens"]["q"]["sha256"] = "different-sequence"
        self.write_records(self.records, self.source_metadata, self.comparisons)
        code, report = self.derive(source_report=False)
        self.assertEqual(code, 1)
        self.assertFalse(report["heldout_selection_pass"])
        self.assertEqual(report["token_sequences_equal_cases"], 1)
        self.assertIsNone(report["original_run"]["accelerator_stats_after_measurement"])

    def test_plan_changed_after_document_is_rejected(self):
        self.plan["selection"] = [2, 3]
        self.plan["excluded_calibration_or_development_indices"] = [0, 1]
        self.save_plan()
        with self.assertRaisesRegex(ValueError, "committed EXCLUDED"):
            self.derive()

    def test_future_declaration_is_rejected(self):
        self.source_metadata["created_utc"] = "2019-01-01T00:00:00+00:00"
        self.write_records(self.records, self.source_metadata, self.comparisons)
        with self.assertRaisesRegex(ValueError, "predate"):
            self.derive()

    def test_foreign_aggregate_report_is_rejected(self):
        self.full_report["overall"]["decision_mismatches"] = 0
        self.source_report.write_text(json.dumps(self.full_report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "overall differs"):
            self.derive()


if __name__ == "__main__":
    unittest.main()
