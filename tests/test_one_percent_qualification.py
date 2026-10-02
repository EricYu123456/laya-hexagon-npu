import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import qualify_one_percent as qualify
from benchmark_fidelity import DATASET, PARQUET, PARQUET_SHA256, REVISION, decision_metrics, json_hash, parse_row, sha256_file, summarize_decisions


class QualificationAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.question = {"type": "noul", "instructions": "Refund?"}
        self.answer = {"type": "noul", "noul": .2, "confidence": .8, "action": {"act_probability": 1.0}}
        self.suite = [{"id": "mixed-length", "state": "example", "questions": {"a": self.question, "b": self.question}}]
        self.write("suite.json", self.suite)
        self.tokens = {"a": {"tokens": 750, "markers": [740, 745], "sha256": "a" * 64},
                       "b": {"tokens": 790, "markers": [780, 785], "sha256": "b" * 64}}
        self.output = {"answers": {"a": self.answer, "b": self.answer}, "usage": {"input_tokens": 1540, "output_tokens": 0}}
        self.runtime = {"packages": {"laya": "0.3.5"}, "laya_source_files": {name: "1" * 64 for name in ("__init__.py", "agent.py", "common.py")}}
        self.models = {"model.safetensors": {"sha256": "c" * 64}}
        self.reference = {"kind": "laya_probe_reference", "backend": "cpu", "module": "laya",
                          "created_utc": "2026-09-29T00:00:00+00:00", "runtime": self.runtime, "model_files": self.models,
                          "suite_sha256": json_hash(self.suite), "suite_file_sha256": sha256_file(self.root / "suite.json"),
                          "records": [{"id": "mixed-length", "input_sha256": json_hash(["example", self.suite[0]["questions"]]),
                                       "tokens": self.tokens, "output": self.output}]}
        self.write("reference.json", self.reference)
        self.graphs = {"768": "7" * 64, "1024": "8" * 64}
        self.plan = {"graph_sha256_by_bucket": self.graphs, "manifest_sha256": "d" * 64,
                     "checkpoint_sha256": "c" * 64, "model_files": self.models,
                     "upstream_laya": qualify.upstream_identity(self.reference), "frozen_utc": "2026-10-02T00:00:00+00:00"}
        manifest = {"model_sha256": self.graphs, "checkpoint_sha256": "c" * 64}
        self.write("candidate-manifest.json", manifest)
        self.plan.update(manifest_evidence="candidate-manifest.json", manifest_sha256=sha256_file(self.root / "candidate-manifest.json"))
        metric = decision_metrics(self.question, self.answer, self.answer)
        self.candidate = copy.deepcopy(self.reference)
        self.candidate.update(kind="laya_probe_comparison", backend="npu", module="laya_npu", created_utc="2026-10-02T00:01:00+00:00",
                              reference_sha256=sha256_file(self.root / "reference.json"), passed=True,
                              overall=summarize_decisions([metric, metric]),
                              candidate_assets={"manifest_sha256": self.plan["manifest_sha256"], "checkpoint_sha256": "c" * 64,
                                                "checkpoint_manifest_verified": True, "manifest": copy.deepcopy(manifest),
                                                "supported_buckets": [768, 1024],
                                                "models": {key: {"sha256": value, "manifest_verified": True} for key, value in self.graphs.items()}},
                              accelerator_stats_after_measurement={"npu_calls": 2, "cpu_fallbacks": 0, "bucket_calls": {"1024": 2},
                                                                   "backend_fingerprint": {"cpu_ep_fallback": False}})
        self.candidate["records"][0].update(tokens_equal=True, reported_tokens_equal=True, npu_execution_verified=True,
                                           expected_bucket_calls={"1024": 2}, accelerator_delta={"npu_calls": 2, "cpu_fallbacks": 0, "bucket_calls": {"1024": 2}},
                                           decisions={"a": metric, "b": metric})
        self.entry = {"name": "fixture", "cases": 1, "decisions": 2, "suite": "suite.json", "reference": "reference.json",
                      "candidate": "candidate.json", "suite_sha256": json_hash(self.suite),
                      "suite_file_sha256": sha256_file(self.root / "suite.json"), "reference_sha256": sha256_file(self.root / "reference.json")}

    def write(self, filename, value):
        (self.root / filename).write_text(json.dumps(value), encoding="utf-8")

    def audit(self):
        self.write("candidate.json", self.candidate)
        return qualify.audit_suite(self.entry, self.plan, self.root)

    def test_valid_mixed_lengths_use_the_collated_bucket(self):
        result, requests = self.audit()
        self.assertTrue(result["passes_one_percent"])
        self.assertEqual(len(requests[0]), 2)
        self.assertEqual(result["accelerator_stats"]["bucket_calls"], {"1024": 2})

    def test_wrong_per_question_bucket_counts_are_rejected(self):
        row = self.candidate["records"][0]
        row["accelerator_delta"]["bucket_calls"] = {"768": 1, "1024": 1}
        with self.assertRaisesRegex(ValueError, "per-request"):
            self.audit()

    def test_forged_pass_or_metric_cannot_qualify(self):
        row = self.candidate["records"][0]
        row["output"]["answers"]["a"]["noul"] = .24
        with self.assertRaisesRegex(ValueError, "Stored decision metrics"):
            self.audit()
        row["decisions"] = {qid: decision_metrics(self.question, self.answer, row["output"]["answers"][qid]) for qid in ("a", "b")}
        self.candidate["overall"] = summarize_decisions(list(row["decisions"].values()))
        result, _ = self.audit()
        self.assertFalse(result["passes_one_percent"])
        self.assertTrue(self.candidate["passed"])

    def test_one_percent_boundary_uses_integer_mismatch_counts(self):
        metric = decision_metrics(self.question, self.answer, self.answer)
        metrics = [copy.deepcopy(metric) for _ in range(100)]
        metrics[0]["decision_mismatch"] = 1
        self.assertTrue(qualify.gate(metrics)["passes_one_percent"])
        self.assertFalse(qualify.gate(metrics[:99])["passes_one_percent"])
        for item in metrics:
            item["total_variation"] = .01001
        self.assertFalse(qualify.gate(metrics)["passes_one_percent"])

    def test_zero_fallback_and_graph_binding_are_mandatory(self):
        original = copy.deepcopy(self.candidate)
        changes = [(lambda c: c["accelerator_stats_after_measurement"].update(cpu_fallbacks=1), "counters"),
                   (lambda c: c["accelerator_stats_after_measurement"]["backend_fingerprint"].update(cpu_ep_fallback=True), "strict execution"),
                   (lambda c: c["candidate_assets"]["models"]["768"].update(sha256="0" * 64), "graph hashes"),
                   (lambda c: c["candidate_assets"].update(checkpoint_manifest_verified=False), "verified graph"),
                   (lambda c: c.update(created_utc="2026-09-30T00:00:00+00:00"), "predates"),
                   (lambda c: c.update(reference_sha256="0" * 64), "frozen inputs")]
        for change, error in changes:
            with self.subTest(error=error):
                self.candidate = copy.deepcopy(original)
                change(self.candidate)
                with self.assertRaisesRegex(ValueError, error):
                    self.audit()

    def test_changed_upstream_or_missing_token_hash_rejected(self):
        self.candidate["runtime"]["laya_source_files"]["agent.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "upstream"):
            self.audit()
        self.candidate["runtime"] = copy.deepcopy(self.runtime)
        self.candidate["records"][0]["tokens"]["a"].pop("sha256")
        with self.assertRaisesRegex(ValueError, "token-ID/marker SHA256"):
            self.audit()

    def test_missing_case_cannot_certify_suite(self):
        self.candidate["records"] = []
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.audit()

    def test_removing_correction_metadata_cannot_bypass_backend_audit(self):
        manifest = qualify.read_json(self.root / "candidate-manifest.json")
        manifest["htp_conv_offset_correction"] = {"1024": {"output_sha256": self.graphs["1024"]}}
        self.write("candidate-manifest.json", manifest)
        digest = sha256_file(self.root / "candidate-manifest.json")
        self.plan["manifest_sha256"] = digest
        self.candidate["candidate_assets"]["manifest_sha256"] = digest
        # A matching claimed SHA alone must not allow an incomplete embedded manifest.
        with self.assertRaisesRegex(ValueError, "embedded manifest"):
            self.audit()

    def test_modified_frozen_manifest_bytes_are_rejected(self):
        self.write("candidate-manifest.json", {"model_sha256": self.graphs, "checkpoint_sha256": "c" * 64, "mask_penalty": -15})
        with self.assertRaisesRegex(ValueError, "manifest evidence hash"):
            self.audit()

    def test_all_suites_gate_separately(self):
        plan = {"suites": [{"name": "large"}, {"name": "small"}], "subsets": {}, "scope": "regression"}
        self.write("qualification-plan.json", plan)
        metric = decision_metrics(self.question, self.answer, self.answer)
        passed = qualify.gate([metric] * 200)
        failed = qualify.gate([{**metric, "decision_mismatch": 1}])
        with patch.object(qualify, "validate_plan", return_value=plan), patch.object(qualify, "audit_suite", side_effect=[(passed, []), (failed, [])]):
            report = qualify.audit(self.root)
        self.assertFalse(report["passed"])

    def test_legacy_inventory_preserves_actual_historical_inputs(self):
        legacy = qualify.read_json(qualify.ROOT / "tests/legacy-fidelity-probes.json")
        originals = qualify.read_json(qualify.ROOT / "accuracy-probes.json")
        example = qualify.read_json(qualify.ROOT / "example.json")
        self.assertEqual((len(legacy), sum(len(p["questions"]) for p in legacy)), (7, 8))
        for probe, original in zip(legacy[:4], originals):
            self.assertEqual(probe["state"], original["state"])
            self.assertEqual(probe["questions"]["test_q"], {"type": "noul", "instructions": original["instructions"]})
        self.assertEqual({k: v for k, v in legacy[4].items() if k != "id"}, example)
        self.assertEqual(legacy[5]["state"], originals[2]["state"])
        self.assertEqual(legacy[5]["questions"]["refund_requested"], example["questions"]["refund_requested"])

    def test_typed_reference_conversion_preserves_original_answers(self):
        rows = [{"id": str(i), "workflow": "fixture", "state": json.dumps(f"state {i}"), "questions": {"a": self.question}}
                for i in range(400)]
        originals = {i: {**parse_row(row, i)[0], "cpu": self.output, "tokens": self.tokens, "cpu_ms": 12.5}
                     for i, row in enumerate(rows)}
        metadata = {"dataset": {"id": DATASET, "revision": REVISION, "file": PARQUET, "sha256": PARQUET_SHA256},
                    "model_files": self.models, "backend": "cpu", "module": "laya", "created_utc": "2026-09-29T00:00:00+00:00",
                    "threads": 4, "runtime": self.runtime}
        suite, reference = qualify.typed_probe_reference(rows, metadata, originals, "e" * 64)
        self.assertEqual(len(suite), 400)
        self.assertEqual(reference["records"][399]["output"], originals[399]["cpu"])
        self.assertEqual(reference["records"][399]["tokens"], originals[399]["tokens"])
        self.assertEqual(reference["provenance"]["source_sha256"], "e" * 64)
        source = self.root / "original.jsonl"
        source.write_text("\n".join(json.dumps(row) for row in [
            {**metadata, "kind": "metadata", "format_version": 1},
            *[{**row, "kind": "reference"} for row in originals.values()]]), encoding="utf-8")
        source_hash = sha256_file(source)
        reference["provenance"]["source_sha256"] = source_hash
        qualify.validate_typed_conversion(suite, reference, source, source_hash)
        changed = copy.deepcopy(reference)
        changed["records"][399]["output"]["answers"]["a"]["noul"] = .21
        with self.assertRaisesRegex(ValueError, "CPU answers/tokens"):
            qualify.validate_typed_conversion(suite, changed, source, source_hash)
        changed = copy.deepcopy(suite)
        changed[0]["state"] = "Changed state"
        with self.assertRaisesRegex(ValueError, "inputs/order"):
            qualify.validate_typed_conversion(changed, reference, source, source_hash)
        originals[399]["input_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "input_sha256"):
            qualify.typed_probe_reference(rows, metadata, originals, "e" * 64)
        del originals[399]
        with self.assertRaisesRegex(ValueError, "400 cases"):
            qualify.typed_probe_reference(rows, metadata, originals, "e" * 64)

    def test_immutable_inventory_covers_every_suite(self):
        inventory = qualify.historical_inventory()
        self.assertEqual([entry["name"] for entry in inventory], list(qualify.SUITE_NAMES))
        self.assertEqual(sum(entry["decisions"] for entry in inventory), 3263)
        for entry in inventory:
            for key in ("suite", "reference"):
                if key in entry:
                    self.assertEqual(sha256_file(qualify.ROOT / entry[key]), entry[key + "_sha256"])

    def test_interrupted_suite_can_resume_without_overwriting_logs(self):
        first = qualify.next_log_path(self.root, "fixture", False)
        self.assertEqual(first.name, "fixture.log")
        first.write_text("Interrupted before final report", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "--resume"):
            qualify.next_log_path(self.root, "fixture", False)
        second = qualify.next_log_path(self.root, "fixture", True)
        self.assertEqual(second.name, "fixture.retry-2.log")
        second.write_text("Second attempt", encoding="utf-8")
        self.assertEqual(qualify.next_log_path(self.root, "fixture", True).name, "fixture.retry-3.log")
        self.assertEqual(first.read_text(encoding="utf-8"), "Interrupted before final report")


if __name__ == "__main__":
    unittest.main()
