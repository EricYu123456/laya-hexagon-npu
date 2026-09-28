#!/usr/bin/env python3
"""Audit completed, predeclared external fidelity comparisons without inference.

The primary gate requires every planned suite AND their aggregate to meet the
unchanged 5% decision-error / 5% mean-TV limits. Gold accuracy is ancillary:
these adapted tasks are not the datasets' original benchmark scores. MASSIVE
translations share semantic IDs and are reported as paired, correlated inputs.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

from benchmark_fidelity import (
    ROOT, decision_metrics, json_hash, sha256_file, summarize_decisions,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def upstream_identity(report):
    runtime = report.get("runtime") or {}
    sources = runtime.get("laya_source_files") or {}
    names = ("__init__.py", "agent.py", "common.py")
    require(all(is_sha256(sources.get(name)) for name in names), "Missing complete upstream Laya source fingerprints")
    version = runtime.get("packages", {}).get("laya")
    require(isinstance(version, str) and version, "Missing upstream Laya package version")
    return {"version": version, "source_sha256": {name: sources[name] for name in names}}


def timestamp(value):
    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(value.tzinfo is not None, "Evidence timestamps must include a timezone")
    return value


def rooted_path(root, name):
    require(isinstance(name, str) and name, "Evidence paths must be nonempty root-relative strings")
    path = Path(name)
    require(not path.is_absolute(), "Evidence paths must be root-relative")
    path = (root / path).resolve()
    require(path.is_relative_to(root.resolve()), "Evidence path escapes the repository")
    return path


def numeric_equal(actual, recorded):
    """Allow only cross-Python floating summation noise, not rounded metrics."""
    if isinstance(actual, dict):
        return (isinstance(recorded, dict) and actual.keys() == recorded.keys()
                and all(numeric_equal(value, recorded[key]) for key, value in actual.items()))
    if isinstance(actual, list):
        return (isinstance(recorded, list) and len(actual) == len(recorded)
                and all(numeric_equal(a, b) for a, b in zip(actual, recorded)))
    if isinstance(actual, float):
        return (type(recorded) in (int, float) and math.isfinite(recorded)
                and math.isclose(actual, recorded, rel_tol=0, abs_tol=1e-12))
    return type(actual) is type(recorded) and actual == recorded


def declaration_commit(root, plan_path):
    """Find an existing Git commit containing these exact plan bytes."""
    relative = plan_path.resolve().relative_to(root.resolve()).as_posix()
    raw = plan_path.read_bytes()
    commits = subprocess.check_output(
        ["git", "-C", str(root), "log", "--format=%H", "--", relative], text=True
    ).splitlines()
    for commit in reversed(commits):
        saved = subprocess.check_output(["git", "-C", str(root), "show", f"{commit}:{relative}"])
        if saved == raw:
            committed = subprocess.check_output(
                ["git", "-C", str(root), "show", "-s", "--format=%cI", commit], text=True
            ).strip()
            return {"commit": commit, "committed_utc": timestamp(committed).isoformat(),
                    "path": relative, "sha256": hashlib.sha256(saved).hexdigest()}
    raise ValueError("The exact evaluation plan must be committed before CPU/NPU inference")


def group_summary(rows, limits):
    metrics = summarize_decisions([row["metrics"] for row in rows])
    passed = (metrics["decisions"] > 0
              and metrics["decision_error_percent"] <= limits["decision_error_percent_max"]
              and metrics["total_variation"]["mean"] <= limits["mean_total_variation_max"])
    gold = {"note": "Ancillary accuracy on adapted tasks, not original dataset benchmark scores.",
            "decisions": len(rows),
            "reference_correct": sum(r["reference_choice"] == r["gold_label"] for r in rows),
            "candidate_correct": sum(r["candidate_choice"] == r["gold_label"] for r in rows)}
    gold["reference_accuracy_percent"] = 100 * gold["reference_correct"] / len(rows)
    gold["candidate_accuracy_percent"] = 100 * gold["candidate_correct"] / len(rows)
    return {"fidelity": metrics, "meets_5_percent_thresholds": passed, "gold_accuracy": gold}


def grouped(rows, key, limits):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return {name: group_summary(items, limits) for name, items in sorted(groups.items())}


def paired_massive(rows, expected_locales):
    if not rows:
        return None
    groups = defaultdict(dict)
    for row in rows:
        key = (row["dataset"], row["semantic_id"])
        require(row["locale"] not in groups[key], "Duplicate MASSIVE semantic ID within one locale")
        groups[key][row["locale"]] = row
    paired = []
    for (dataset, semantic_id), translations in sorted(groups.items()):
        require(set(translations) == expected_locales, "MASSIVE paired locale coverage is incomplete")
        require(len({r["gold_label"] for r in translations.values()}) == 1,
                "MASSIVE paired gold scenarios differ")
        require(len({r["source_intent"] for r in translations.values()}) == 1,
                "MASSIVE paired source intents differ")
        paired.append({"dataset": dataset, "semantic_id": semantic_id,
                       "any_decision_mismatch": any(r["metrics"]["decision_mismatch"] for r in translations.values()),
                       "maximum_total_variation": max(r["metrics"]["total_variation"] for r in translations.values()),
                       "by_locale": {locale: {"id": r["id"], "reference_choice": r["reference_choice"],
                                               "candidate_choice": r["candidate_choice"],
                                               "decision_mismatch": r["metrics"]["decision_mismatch"],
                                               "total_variation": r["metrics"]["total_variation"]}
                                     for locale, r in sorted(translations.items())}})
    contrasts = {}
    locales = sorted(expected_locales)
    for i, left in enumerate(locales):
        for right in locales[i + 1:]:
            pairs = [(p["by_locale"][left], p["by_locale"][right]) for p in paired]
            contrasts[f"{left} / {right}"] = {
                "semantic_ids": len(pairs),
                "both_match": sum(not x["decision_mismatch"] and not y["decision_mismatch"] for x, y in pairs),
                "both_mismatch": sum(x["decision_mismatch"] and y["decision_mismatch"] for x, y in pairs),
                "only_first_mismatches": sum(x["decision_mismatch"] and not y["decision_mismatch"] for x, y in pairs),
                "only_second_mismatches": sum(not x["decision_mismatch"] and y["decision_mismatch"] for x, y in pairs),
                "mean_tv_first_minus_second": statistics.mean(x["total_variation"] - y["total_variation"] for x, y in pairs),
                "reference_choices_differ": sum(x["reference_choice"] != y["reference_choice"] for x, y in pairs),
                "candidate_choices_differ": sum(x["candidate_choice"] != y["candidate_choice"] for x, y in pairs),
            }
    return {"note": "Translations are correlated measurements of the same semantic IDs; no independent-sample confidence claim is made.",
            "semantic_ids": len(paired), "decisions": len(rows), "locales": locales,
            "semantic_ids_with_any_mismatch": sum(p["any_decision_mismatch"] for p in paired),
            "pairwise_locale_comparisons": contrasts, "records": paired}


def audit_suite(entry, plan, root, proof, limits):
    paths = {key: rooted_path(root, entry[key]) for key in ("suite", "reference", "candidate")}
    suite, ref, candidate = (read_json(paths[key]) for key in ("suite", "reference", "candidate"))
    require(type(entry["cases"]) is int and entry["cases"] > 0, "Invalid planned suite count")
    require(isinstance(suite, list) and len(suite) == entry["cases"], "Suite case count differs from plan")
    require(json_hash(suite) == entry["suite_sha256"], "Suite semantic hash differs from plan")
    require(sha256_file(paths["suite"]) == entry["suite_file_sha256"], "Suite file hash differs from plan")
    require((ref.get("kind"), ref.get("backend"), ref.get("module")) == ("laya_probe_reference", "cpu", "laya"),
            "Reference must use unchanged CPU Laya")
    require((candidate.get("kind"), candidate.get("backend"), candidate.get("module"))
            == ("laya_probe_comparison", "npu", "laya_npu"), "Candidate must use strict NPU Laya")
    for report in (ref, candidate):
        require(report.get("suite_sha256") == entry["suite_sha256"]
                and report.get("suite_file_sha256") == entry["suite_file_sha256"], "Report belongs to another suite")
        require(timestamp(report["created_utc"]) >= timestamp(proof["committed_utc"]),
                "Inference preceded the committed declaration")
    require(candidate.get("reference_sha256") == sha256_file(paths["reference"]), "Candidate reference hash differs")
    require(candidate.get("model_files") == ref.get("model_files") and ref.get("model_files"), "Model fingerprints differ")
    upstream = upstream_identity(ref)
    require(upstream_identity(candidate) == upstream, "CPU/NPU upstream Laya source or version differs")
    assets = candidate.get("candidate_assets") or {}
    graphs = {str(k): v.get("sha256") for k, v in assets.get("models", {}).items()}
    require(graphs == plan["graph_sha256_by_bucket"], "Candidate graph hashes differ from frozen plan")
    require(all(v.get("manifest_verified") is True for v in assets["models"].values()), "Unverified candidate graph")
    require(assets.get("checkpoint_manifest_verified") is True, "Unverified candidate checkpoint")
    require(assets.get("manifest", {}).get("model_sha256") == graphs, "Manifest graph binding differs")
    if plan.get("manifest_sha256"):
        require(assets.get("manifest_sha256") == plan["manifest_sha256"], "Candidate manifest differs from plan")
    if plan.get("checkpoint_sha256"):
        require(assets.get("checkpoint_sha256") == plan["checkpoint_sha256"], "Candidate checkpoint differs from plan")
        require(ref["model_files"].get("model.safetensors", {}).get("sha256") == plan["checkpoint_sha256"],
                "Reference checkpoint differs from plan")
    buckets = sorted(int(b) for b in graphs)
    require(assets.get("supported_buckets") == buckets, "Candidate bucket metadata differs")
    ids = [p["id"] for p in suite]
    require(len(set(ids)) == len(ids), "Duplicate suite IDs")
    require(ids == [r["id"] for r in ref["records"]] == [r["id"] for r in candidate["records"]],
            "Reference/candidate coverage or order differs from the complete suite")
    rows, bucket_calls, fixed_question = [], defaultdict(int), None
    for probe, original, actual in zip(suite, ref["records"], candidate["records"]):
        require(set(probe["questions"]) == {"route"}, "Each external probe must contain one route question")
        question = probe["questions"]["route"]
        require(question.get("type") == "choice", "External route questions must use choice")
        criteria = question.get("criteria")
        require(isinstance(criteria, (dict, list)) and len(criteria) >= 2, "Invalid fixed choice criteria")
        labels = list(criteria)
        require(all(isinstance(k, str) and k for k in labels) and len(set(labels)) == len(labels), "Invalid choice labels")
        question_hash = json_hash(question)
        fixed_question = fixed_question or question_hash
        require(fixed_question == question_hash, "Question/ordered candidates change between rows of one suite")
        metadata = probe["metadata"]
        require(all(key in metadata for key in ("dataset", "locale", "source_row_id", "source_intent", "gold_label", "semantic_id")),
                "Missing source/grouping metadata")
        require(metadata["gold_label"] in labels, "Gold label is absent from the fixed criteria")
        require(original["input_sha256"] == actual["input_sha256"] == json_hash([probe["state"], probe["questions"]]),
                "Input identity mismatch")
        require(set(original["tokens"]) == {"route"} and original["tokens"] == actual["tokens"], "Token/marker identities differ")
        tokens = original["tokens"]["route"]
        require(is_sha256(tokens.get("sha256")), "Missing token-ID/marker SHA256 fingerprint")
        require(type(tokens["tokens"]) is int and 1 <= tokens["tokens"] <= 1024, "Invalid original token budget")
        markers = tokens["markers"]
        require(len(markers) == len(labels) and markers == sorted(set(markers))
                and all(type(m) is int and 0 <= m < tokens["tokens"] for m in markers), "Choice markers are incomplete")
        require(actual.get("tokens_equal") is True and actual.get("reported_tokens_equal") is True, "Input equality flags failed")
        require(original["output"]["usage"] == actual["output"]["usage"]
                and original["output"]["usage"]["input_tokens"] == tokens["tokens"], "Reported token usage differs")
        require(set(original["output"]["answers"]) == set(actual["output"]["answers"]) == {"route"}, "Answer coverage differs")
        bucket = str(next((b for b in buckets if b >= tokens["tokens"]), 0))
        expected_calls = {bucket: 1}
        require(actual.get("npu_execution_verified") is True
                and actual.get("expected_bucket_calls") == expected_calls
                and actual.get("accelerator_delta") == {"npu_calls": 1, "cpu_fallbacks": 0, "bucket_calls": expected_calls},
                "Missing actual per-decision NPU execution proof")
        bucket_calls[bucket] += 1
        ref_answer, npu_answer = original["output"]["answers"]["route"], actual["output"]["answers"]["route"]
        metrics = decision_metrics(question, ref_answer, npu_answer)
        require(numeric_equal({"route": metrics}, actual.get("decisions")), "Stored decision metrics differ from recomputation")
        rows.append({**metadata, "id": probe["id"], "suite": entry["name"], "tokens": tokens["tokens"],
                     "reference_choice": ref_answer["choice"], "candidate_choice": npu_answer["choice"], "metrics": metrics})
    require(len({r["dataset"] for r in rows}) == len({r["locale"] for r in rows}) == 1,
            "A suite must have one dataset and locale")
    stats = candidate.get("accelerator_stats_after_measurement") or {}
    require(stats.get("npu_calls") == len(rows) and stats.get("bucket_calls") == dict(bucket_calls)
            and stats.get("cpu_fallbacks") == 0, "Aggregate NPU counters differ from complete records")
    require(stats.get("backend_fingerprint", {}).get("cpu_ep_fallback") is False, "Backend does not confirm strict execution")
    if "htp_conv_offset_correction" in assets["manifest"]:
        from npu.fidelity_runtime import correction_runtime_requirements, validate_correction_backend
        requirements = correction_runtime_requirements(assets["manifest"], buckets)
        for bucket in bucket_calls:
            if int(bucket) in requirements:
                correction = assets["manifest"]["htp_conv_offset_correction"][bucket]
                require(correction["output_sha256"] == graphs[bucket], "Correction output graph binding differs")
                require(int(bucket) in stats.get("correction_backend_verified_buckets", []),
                        "Used corrected bucket lacks backend verification proof")
                validate_correction_backend(requirements[int(bucket)], stats["backend_fingerprint"], int(bucket))
    summary = group_summary(rows, limits)
    require(numeric_equal(summary["fidelity"], candidate.get("overall")), "Stored aggregate metrics differ from recomputation")
    require(candidate.get("passed") is summary["meets_5_percent_thresholds"], "Candidate pass flag differs from unchanged fidelity gates")
    summary.update({"name": entry["name"], "dataset": rows[0]["dataset"], "locale": rows[0]["locale"],
                    "evidence": {key: {"path": entry[key], "sha256": sha256_file(path)} for key, path in paths.items()},
                    "tokens": {"min": min(r["tokens"] for r in rows), "max": max(r["tokens"] for r in rows),
                               "mean": statistics.mean(r["tokens"] for r in rows), "bucket_calls": dict(bucket_calls)},
                    "accelerator_stats_original_run": stats,
                    "reference_runtime": ref.get("runtime"), "candidate_runtime": candidate.get("runtime"),
                    "upstream_laya": upstream,
                    "model_files": ref["model_files"], "candidate_manifest_sha256": assets["manifest_sha256"]})
    return rows, summary


def summarize_plan(plan_path, root=ROOT):
    root = Path(root).resolve()
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    require(isinstance(plan.get("kind"), str) and "external" in plan["kind"], "Expected an external evaluation plan")
    require(plan.get("suites") and isinstance(plan.get("graph_sha256_by_bucket"), dict), "Missing planned suites or frozen graphs")
    limits = plan.get("criteria", {"decision_error_percent_max": 5.0, "mean_total_variation_max": .05})
    require(limits.get("decision_error_percent_max") == 5 and limits.get("mean_total_variation_max") == .05,
            "External summary retains the original 5% decision / mean-TV limits")
    proof = declaration_commit(root, plan_path)
    require(timestamp(plan["created_utc"]) <= timestamp(proof["committed_utc"]), "Plan timestamp follows its Git declaration")
    source_lock = None
    if plan.get("source_lock"):
        source_path = rooted_path(root, plan["source_lock"])
        require(sha256_file(source_path) == plan.get("source_lock_sha256"), "Source lock hash differs from plan")
        source_lock = {"path": plan["source_lock"], "sha256": sha256_file(source_path),
                       "contents": read_json(source_path),
                       "note": "The frozen source lock is verified. Raw source downloads are not required or fetched by this offline report audit."}
    names = [entry["name"] for entry in plan["suites"]]
    require(len(set(names)) == len(names), "Duplicate planned suite names")
    evidence_paths = [entry[key] for entry in plan["suites"] for key in ("suite", "reference", "candidate")]
    require(len(set(evidence_paths)) == len(evidence_paths), "Planned suites reuse evidence files")
    rows, summaries, massive_rows, massive_locales = [], {}, [], set()
    for entry in plan["suites"]:
        items, summary = audit_suite(entry, plan, root, proof, limits)
        rows.extend(items)
        summaries[entry["name"]] = summary
        if entry["name"].lower().startswith("massive-"):
            massive_rows.extend(items)
            massive_locales.add(items[0]["locale"])
    require(len({row["id"] for row in rows}) == len(rows), "Probe IDs overlap between planned suites")
    fingerprints = {json_hash(summary["model_files"]) for summary in summaries.values()}
    require(len(fingerprints) == 1, "Original checkpoint/tokenizer fingerprints differ between suites")
    upstream_sources = {json_hash(summary["upstream_laya"]) for summary in summaries.values()}
    require(len(upstream_sources) == 1, "Upstream Laya source/version changed between suites")
    backends = {json_hash(summary["accelerator_stats_original_run"]["backend_fingerprint"]) for summary in summaries.values()}
    require(len(backends) == 1, "NPU backend fingerprint changed between suites")
    require(len({summary["candidate_manifest_sha256"] for summary in summaries.values()}) == 1,
            "NPU manifest changed between suites")
    overall = group_summary(rows, limits)
    dataset_groups = defaultdict(list)
    for row in rows:
        dataset_groups[row["dataset"]].append(row)
    datasets = {dataset: {**group_summary(items, limits), "by_locale": grouped(items, "locale", limits),
                          "by_source_intent": grouped(items, "source_intent", limits),
                          "by_target_label": grouped(items, "gold_label", limits)}
                for dataset, items in sorted(dataset_groups.items())}
    passed = overall["meets_5_percent_thresholds"] and all(s["meets_5_percent_thresholds"] for s in summaries.values())
    return {"kind": "laya_external_fidelity_summary", "created_utc": datetime.now(timezone.utc).isoformat(),
            "plan": {"path": plan_path.relative_to(root).as_posix(), "sha256": sha256_file(plan_path),
                     "declaration": proof, "contents": plan},
            "source_lock": source_lock,
            "criteria": limits, "passed": passed, "overall": overall, "suites": summaries, "datasets": datasets,
            "massive_paired": paired_massive(massive_rows, massive_locales),
            "notes": ["Primary passed requires all planned suites and the aggregate to pass; no failed suite is hidden by pooling.",
                      "Intent/target-label subgroups are descriptive diagnostics, not additional acceptance gates.",
                      "Gold accuracy describes adapted tasks: restricted BANKING77 intents, CLINC domains and MASSIVE scenarios; it is not an original benchmark score.",
                      "MASSIVE translations are paired by semantic ID; the decision count is not an independent semantic sample count.",
                      "Token truncation integrity is documented separately by preflight_external.py; this audit verifies recorded token-ID/marker fingerprints without retokenizing.",
                      "CPU and NPU timings come from different hosts and are not a speed comparison.",
                      "Raw model outputs remain in the hashed reference/candidate evidence files; this tool performs no inference."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Use a fresh output path; existing evidence is never overwritten")
    report = summarize_plan(args.plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": report["passed"], "decisions": report["overall"]["fidelity"]["decisions"],
                      "suites": {name: {"passed": s["meets_5_percent_thresholds"],
                                        "decision_error_percent": s["fidelity"]["decision_error_percent"],
                                        "mean_total_variation": s["fidelity"]["total_variation"]["mean"]}
                                 for name, s in report["suites"].items()}}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
