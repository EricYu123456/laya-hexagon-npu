"""Offline external-evidence audits using fabricated outputs, never models."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark_fidelity import decision_metrics, json_hash, sha256_file, summarize_decisions
import summarize_external as external


class ExternalSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plan_path = self.root / "plan.json"
        self.graphs = {"768": "a" * 64, "1024": "b" * 64}
        self.weight = "c" * 64
        self.manifest = "d" * 64
        self.question = {"type": "choice", "instructions": "Choose the matching domain.", "criteria": ["alpha", "beta"]}
        self.plan = {"kind": "laya_external_fidelity_plan", "created_utc": "2026-09-29T00:00:00+00:00",
                     "criteria": {"decision_error_percent_max": 5.0, "mean_total_variation_max": .05},
                     "graph_sha256_by_bucket": self.graphs, "checkpoint_sha256": self.weight,
                     "manifest_sha256": self.manifest, "suites": [],
                     "source_lock": "sources.json"}
        self.write("sources.json", {"corpus": {"sha256": "e" * 64}})
        self.plan["source_lock_sha256"] = sha256_file(self.root / "sources.json")
        for name, count, dataset, locale in [
            ("banking77-card8", 1, "banking77", "en"),
            ("clinc150-domain10", 40, "clinc150", "en"),
            ("massive-en-US", 20, "massive", "en-US"),
            ("massive-zh-TW", 20, "massive", "zh-TW"),
            ("massive-zh-CN", 20, "massive", "zh-CN"),
        ]:
            self.make_suite(name, count, dataset, locale)
        self.write("plan.json", self.plan)
        self.proof = {"commit": "f" * 40, "committed_utc": "2026-09-29T00:01:00+00:00",
                      "path": "plan.json", "sha256": sha256_file(self.plan_path)}

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def read(self, name):
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def make_suite(self, name, count, dataset, locale):
        suite = [{"id": f"{name}-{i}", "state": f"User message {i}",
                  "questions": {"route": copy.deepcopy(self.question)},
                  "metadata": {"dataset": dataset, "locale": locale, "source_row_id": str(i),
                               "source_intent": "intent", "gold_label": "alpha", "semantic_id": f"{dataset}-{i}"}}
                 for i in range(count)]
        suite_name, ref_name, candidate_name = f"{name}-suite.json", f"{name}-reference.json", f"{name}-npu.json"
        self.write(suite_name, suite)
        token = {"route": {"sha256": "1" * 64, "tokens": 32, "markers": [3, 8]}}
        answer = {"type": "choice", "choice": "alpha", "probabilities": {"alpha": .8, "beta": .2},
                  "confidence": .8, "action": {"act_probability": 1.0}}
        ref = {"kind": "laya_probe_reference", "backend": "cpu", "module": "laya",
               "created_utc": "2026-09-29T00:02:00+00:00", "suite_sha256": json_hash(suite),
               "suite_file_sha256": sha256_file(self.root / suite_name),
               "model_files": {"model.safetensors": {"sha256": self.weight}}, "records": [],
               "runtime": {"packages": {"laya": "0.3.5"},
                           "laya_source_files": {name: "9" * 64 for name in ("__init__.py", "agent.py", "common.py")}}}
        for probe in suite:
            ref["records"].append({"id": probe["id"], "input_sha256": json_hash([probe["state"], probe["questions"]]),
                                   "tokens": copy.deepcopy(token),
                                   "output": {"model": "laya-rl-agent", "answers": {"route": copy.deepcopy(answer)},
                                              "usage": {"input_tokens": 32, "output_tokens": 0}}})
        self.write(ref_name, ref)
        candidate = copy.deepcopy(ref)
        candidate.update(kind="laya_probe_comparison", backend="npu", module="laya_npu",
                         reference_sha256=sha256_file(self.root / ref_name),
                         created_utc="2026-09-29T00:03:00+00:00")
        candidate["candidate_assets"] = {
            "models": {k: {"sha256": v, "manifest_verified": True} for k, v in self.graphs.items()},
            "manifest": {"model_sha256": self.graphs}, "manifest_sha256": self.manifest,
            "checkpoint_manifest_verified": True, "checkpoint_sha256": self.weight,
            "supported_buckets": [768, 1024]}
        for row in candidate["records"]:
            row.update(tokens_equal=True, reported_tokens_equal=True, npu_execution_verified=True,
                       expected_bucket_calls={"768": 1},
                       accelerator_delta={"npu_calls": 1, "cpu_fallbacks": 0, "bucket_calls": {"768": 1}})
            row["decisions"] = {"route": decision_metrics(self.question, answer, answer)}
        candidate["overall"] = summarize_decisions([r["decisions"]["route"] for r in candidate["records"]])
        candidate["passed"] = True
        candidate["accelerator_stats_after_measurement"] = {
            "npu_calls": count, "bucket_calls": {"768": count}, "cpu_fallbacks": 0,
            "backend_fingerprint": {"cpu_ep_fallback": False, "backend_sha256": "same"}}
        self.write(candidate_name, candidate)
        self.plan["suites"].append({"name": name, "cases": count, "suite": suite_name,
                                    "reference": ref_name, "candidate": candidate_name,
                                    "suite_sha256": json_hash(suite), "suite_file_sha256": sha256_file(self.root / suite_name)})

    def summarize(self):
        with patch.object(external, "declaration_commit", return_value=self.proof):
            return external.summarize_plan(self.plan_path, self.root)

    def test_complete_reports_recompute_all_metrics_and_pair_translations(self):
        summary = self.summarize()
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["overall"]["fidelity"]["decisions"], 101)
        self.assertEqual(summary["massive_paired"]["semantic_ids"], 20)
        self.assertEqual(summary["massive_paired"]["decisions"], 60)
        self.assertEqual(len(summary["massive_paired"]["pairwise_locale_comparisons"]), 3)
        self.assertEqual(summary["datasets"]["clinc150"]["by_source_intent"]["intent"]["fidelity"]["decisions"], 40)

    def test_pooling_cannot_hide_one_failed_suite(self):
        entry = self.plan["suites"][0]
        candidate = self.read(entry["candidate"])
        ref = self.read(entry["reference"])
        candidate["records"][0]["output"]["answers"]["route"].update(choice="beta", probabilities={"alpha": .2, "beta": .8})
        metric = decision_metrics(self.question, ref["records"][0]["output"]["answers"]["route"],
                                  candidate["records"][0]["output"]["answers"]["route"])
        candidate["records"][0]["decisions"] = {"route": metric}
        candidate["overall"] = summarize_decisions([metric])
        candidate["passed"] = False
        self.write(entry["candidate"], candidate)
        summary = self.summarize()
        self.assertTrue(summary["overall"]["meets_5_percent_thresholds"])
        self.assertFalse(summary["passed"])
        self.assertFalse(summary["suites"]["banking77-card8"]["meets_5_percent_thresholds"])

    def test_tampered_metric_is_not_trusted(self):
        entry = self.plan["suites"][0]
        candidate = self.read(entry["candidate"])
        candidate["records"][0]["decisions"]["route"]["total_variation"] = .01
        self.write(entry["candidate"], candidate)
        with self.assertRaisesRegex(ValueError, "Stored decision metrics"):
            self.summarize()

    def test_changed_graph_or_reference_cannot_reuse_report(self):
        entry = self.plan["suites"][0]
        candidate = self.read(entry["candidate"])
        for mutate, pattern in [(lambda c: c["candidate_assets"]["models"]["768"].update(sha256="changed"), "graph hashes"),
                                (lambda c: c.update(reference_sha256="changed"), "reference hash")]:
            with self.subTest(pattern=pattern):
                changed = copy.deepcopy(candidate)
                mutate(changed)
                self.write(entry["candidate"], changed)
                with self.assertRaisesRegex(ValueError, pattern):
                    self.summarize()

    def test_missing_case_and_faked_counter_are_rejected(self):
        entry = self.plan["suites"][0]
        candidate = self.read(entry["candidate"])
        for mutate, pattern in [(lambda c: c["records"].clear(), "coverage"),
                                (lambda c: c["records"][0]["accelerator_delta"].update(npu_calls=0), "per-decision")]:
            with self.subTest(pattern=pattern):
                changed = copy.deepcopy(candidate)
                mutate(changed)
                self.write(entry["candidate"], changed)
                with self.assertRaisesRegex(ValueError, pattern):
                    self.summarize()

    def test_plan_must_precede_reference_capture(self):
        self.proof["committed_utc"] = "2026-09-29T00:02:30+00:00"
        with self.assertRaisesRegex(ValueError, "preceded the committed"):
            self.summarize()

    def test_changed_suite_or_source_lock_rejected(self):
        self.write("sources.json", {"changed": True})
        with self.assertRaisesRegex(ValueError, "Source lock hash"):
            self.summarize()

    def test_pairing_rejects_missing_or_duplicate_semantic_locale(self):
        base = {"dataset": "massive", "semantic_id": "same", "locale": "en-US", "gold_label": "alpha",
                "source_intent": "intent", "id": "one", "metrics": {"decision_mismatch": 0, "total_variation": 0},
                "reference_choice": "alpha", "candidate_choice": "alpha"}
        with self.assertRaisesRegex(ValueError, "coverage"):
            external.paired_massive([base], {"en-US", "zh-TW"})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            external.paired_massive([base, base], {"en-US"})

    def test_numeric_tolerance_is_only_for_float_summation_noise(self):
        self.assertTrue(external.numeric_equal({"tv": .2}, {"tv": .2000000000000001}))
        self.assertFalse(external.numeric_equal({"tv": .2}, {"tv": .20001}))
        self.assertFalse(external.numeric_equal({"count": 1}, {"count": True}))

    def test_removed_token_hash_from_both_reports_is_rejected(self):
        entry = self.plan["suites"][0]
        ref, candidate = self.read(entry["reference"]), self.read(entry["candidate"])
        for report in (ref, candidate):
            del report["records"][0]["tokens"]["route"]["sha256"]
        self.write(entry["reference"], ref)
        candidate["reference_sha256"] = sha256_file(self.root / entry["reference"])
        self.write(entry["candidate"], candidate)
        with self.assertRaisesRegex(ValueError, "token-ID/marker SHA256"):
            self.summarize()

    def test_missing_or_different_upstream_implementation_rejected(self):
        entry = self.plan["suites"][0]
        candidate = self.read(entry["candidate"])
        for mutate, pattern in [
            (lambda c: c["runtime"]["laya_source_files"].pop("agent.py"), "complete upstream"),
            (lambda c: c["runtime"]["laya_source_files"].update({"agent.py": "8" * 64}), "CPU/NPU upstream"),
            (lambda c: c["runtime"]["packages"].update(laya="different"), "CPU/NPU upstream"),
        ]:
            with self.subTest(pattern=pattern):
                changed = copy.deepcopy(candidate)
                mutate(changed)
                self.write(entry["candidate"], changed)
                with self.assertRaisesRegex(ValueError, pattern):
                    self.summarize()

    def test_cpu_and_npu_agreeing_on_changed_source_in_one_suite_still_rejected(self):
        entry = self.plan["suites"][0]
        ref, candidate = self.read(entry["reference"]), self.read(entry["candidate"])
        for report in (ref, candidate):
            report["runtime"]["laya_source_files"]["common.py"] = "8" * 64
        self.write(entry["reference"], ref)
        candidate["reference_sha256"] = sha256_file(self.root / entry["reference"])
        self.write(entry["candidate"], candidate)
        with self.assertRaisesRegex(ValueError, "changed between suites"):
            self.summarize()

    def test_recorded_correction_backend_and_used_bucket_proof_must_agree(self):
        entry = self.plan["suites"][0]
        candidate = self.read(entry["candidate"])
        calibrated = {"onnxruntime": "1.30.0", "onnxruntime_qnn": "2.5.0", "htp_arch": 68,
                      "backend_sha256": "7" * 64, "cpu_ep_fallback": False}
        candidate["candidate_assets"]["manifest"]["htp_conv_offset_correction"] = {
            "768": {"runtime": calibrated, "output_sha256": self.graphs["768"]}}
        candidate["accelerator_stats_after_measurement"]["backend_fingerprint"] = {**calibrated, "backend_sha256": "6" * 64}
        candidate["accelerator_stats_after_measurement"]["correction_backend_verified_buckets"] = [768]
        self.write(entry["candidate"], candidate)
        with self.assertRaisesRegex(RuntimeError, "correction backend mismatch"):
            self.summarize()
        candidate["accelerator_stats_after_measurement"]["backend_fingerprint"] = calibrated
        candidate["accelerator_stats_after_measurement"]["correction_backend_verified_buckets"] = []
        self.write(entry["candidate"], candidate)
        with self.assertRaisesRegex(ValueError, "lacks backend verification"):
            self.summarize()


if __name__ == "__main__":
    unittest.main()
