#!/usr/bin/env python3
"""Freeze, run and independently audit all historical Laya fidelity suites.

Prepare only after choosing the candidate; the plan locks checkpoint, graph,
corpus, original CPU outputs, and harness bytes before any validation inference.
All historical datasets remain regression evidence, not a new independent test.
The historical 5% report flags are ignored: this auditor recomputes 1% gates.

  python benchmark_probes.py --suite tests/legacy-fidelity-probes.json \
    --backend cpu --output .work/legacy-reference.json
  python qualify_one_percent.py prepare --output-dir .work/qualification \
    --dataset-path .work/dataset/all/test-00000-of-00001.parquet \
    --legacy-reference .work/legacy-reference.json --manifest npu/fidelity/manifest.json
  python qualify_one_percent.py run --output-dir .work/qualification
  python qualify_one_percent.py audit --output-dir .work/qualification

Run on the Pi after sourcing npu/env.sh; reference capture may use WSL. A failed
suite does not skip later suites. --resume only skips already completed reports;
all reports are checked against the same frozen plan before success is reported.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from benchmark_fidelity import (
    ROOT, PARQUET_SHA256, check_reference_metadata, decision_metrics, json_hash,
    load_dataset, parse_row, read_jsonl, sha256_file, summarize_decisions,
    validate_reference_row,
)
from summarize_external import is_sha256, numeric_equal, require, upstream_identity


THRESHOLDS = {"decision_mismatch_max": .01, "mean_total_variation_max": .01}
EXTERNAL_NAMES = ("banking77-card8", "clinc150-domain10", "massive-en-US", "massive-zh-TW", "massive-zh-CN")
SUITE_NAMES = ("typed-decisions-full", "supplementary-development", "long-input", *EXTERNAL_NAMES, "legacy")
EXPECTED = {"typed-decisions-full": (400, 2000), "supplementary-development": (10, 15),
            "long-input": (16, 80), "banking77-card8": (320, 320), "clinc150-domain10": (300, 300),
            "massive-en-US": (180, 180), "massive-zh-TW": (180, 180), "massive-zh-CN": (180, 180),
            "legacy": (7, 8)}
HARNESS = ("qualify_one_percent.py", "benchmark_probes.py", "benchmark_fidelity.py", "summarize_external.py",
           "laya_npu.py", "npu/fidelity_runtime.py")
INVENTORY_PATH = ROOT / "reports/evaluation-plans/one-percent-historical-inventory.json"
INVENTORY_SHA256 = "bcb526b9a1fda46b5f296bc905f7a4de8e15d8a2fb02d4e80dd94b0c5be3eee8"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def historical_inventory():
    require(sha256_file(INVENTORY_PATH) == INVENTORY_SHA256, "Historical inventory changed")
    return read_json(INVENTORY_PATH)["suites"]


def gate(metrics):
    summary = summarize_decisions(metrics)
    passed = (summary["decisions"] > 0
              and summary["decision_mismatches"] * 100 <= summary["decisions"]
              and summary["total_variation"]["mean"] <= .01)
    return {"fidelity": summary, "passes_one_percent": passed}


def bucket_calls(tokens, buckets):
    maximum = max(t["tokens"] for t in tokens.values())
    bucket = next((b for b in buckets if b >= maximum), None)
    require(bucket is not None, "Input exceeds frozen graph capacity")
    return {str(bucket): len(tokens)}


def validate_tokens(probe, tokens, output):
    require(set(tokens) == set(probe["questions"]) == set(output["answers"]), "Token/answer question coverage differs")
    for qid, question in probe["questions"].items():
        token = tokens[qid]
        count, markers = token.get("tokens"), token.get("markers")
        require(is_sha256(token.get("sha256")), "Missing token-ID/marker SHA256")
        require(type(count) is int and probe.get("min_sequence_tokens", 1) <= count <= 1024, "Invalid original token budget")
        size = 2 if question["type"] == "noul" else len(question["criteria"])
        require(isinstance(markers, list) and len(markers) == size
                and markers == sorted(set(markers))
                and all(type(m) is int and 0 <= m < count for m in markers), "Invalid original marker positions")
    require(output["usage"]["input_tokens"] == sum(t["tokens"] for t in tokens.values()), "Reported input token usage differs")


def validate_reference(suite, reference, suite_path):
    require((reference.get("kind"), reference.get("backend"), reference.get("module"))
            == ("laya_probe_reference", "cpu", "laya"), "Reference must use unchanged CPU Laya")
    require(reference.get("suite_sha256") == json_hash(suite)
            and reference.get("suite_file_sha256") == sha256_file(suite_path), "Reference suite hashes differ")
    require([p["id"] for p in suite] == [r["id"] for r in reference["records"]], "Reference coverage/order differs")
    require(len({p["id"] for p in suite}) == len(suite), "Duplicate suite IDs")
    upstream_identity(reference)
    for probe, original in zip(suite, reference["records"]):
        require(original["input_sha256"] == json_hash([probe["state"], probe["questions"]]), "Reference input identity differs")
        validate_tokens(probe, original["tokens"], original["output"])
        for qid, question in probe["questions"].items():
            answer = original["output"]["answers"][qid]
            decision_metrics(question, answer, answer)


def typed_probe_reference(rows, metadata, originals, source_sha256):
    """Lossless format conversion; never changes original outputs or token IDs."""
    check_reference_metadata(metadata, metadata.get("model_files"))
    require(len(rows) == 400 and set(originals) == set(range(400)), "Typed reference must cover all 400 cases")
    suite, records = [], []
    for index, row in enumerate(rows):
        identity, state, questions = parse_row(row, index)
        baseline = originals[index]
        validate_reference_row(baseline, identity)
        probe = {"id": f"typed-{index:03d}-{identity['id']}", "state": state, "questions": questions,
                 "metadata": {"index": index, "workflow": identity["workflow"]}}
        suite.append(probe)
        records.append({"id": probe["id"], "input_sha256": baseline["input_sha256"],
                        "tokens": baseline["tokens"], "output": baseline["cpu"], "elapsed_ms": baseline["cpu_ms"]})
    reference = {"kind": "laya_probe_reference", "backend": "cpu", "module": "laya",
                 "created_utc": metadata["created_utc"], "threads": metadata["threads"],
                 "runtime": metadata["runtime"], "model_files": metadata["model_files"],
                 "records": records, "suite_sha256": json_hash(suite),
                 "provenance": {"operation": "lossless original JSONL format conversion", "source_sha256": source_sha256}}
    return suite, reference


def validate_typed_conversion(suite, reference, original_path, expected_sha256):
    """Independently rebind derived probe evidence to the immutable CPU JSONL."""
    require(sha256_file(original_path) == expected_sha256, "Historical typed CPU source changed")
    metadata, originals = read_jsonl(original_path)
    check_reference_metadata(metadata, metadata.get("model_files"))
    require(len(suite) == len(reference["records"]) == 400 and set(originals) == set(range(400)),
            "Converted typed reference must cover all 400 original cases")
    require(reference.get("provenance") == {
        "operation": "lossless original JSONL format conversion", "source_sha256": expected_sha256},
        "Converted typed reference provenance differs")
    for key in ("created_utc", "threads", "runtime", "model_files"):
        require(reference.get(key) == metadata[key], f"Converted typed reference changed original {key}")
    for index, (probe, record) in enumerate(zip(suite, reference["records"])):
        original = originals[index]
        expected_id = f"typed-{index:03d}-{original['id']}"
        require(probe["id"] == expected_id
                and probe.get("metadata") == {"index": index, "workflow": original["workflow"]}
                and json_hash([probe["state"], probe["questions"]]) == original["input_sha256"],
                "Converted typed inputs/order differ from immutable original")
        require(record == {"id": expected_id, "input_sha256": original["input_sha256"],
                           "tokens": original["tokens"], "output": original["cpu"], "elapsed_ms": original["cpu_ms"]},
                "Converted typed CPU answers/tokens differ from immutable original")


def prepare(args):
    output = args.output_dir.resolve()
    require(not output.exists(), "Prepare requires a new output directory")
    inventory = historical_inventory()
    for entry in inventory:
        for key in ("suite", "reference"):
            if key in entry:
                require(sha256_file(ROOT / entry[key]) == entry[f"{key}_sha256"], f"Historical {key} changed for {entry['name']}")
    rows = load_dataset(args.dataset_path)
    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    require(isinstance(manifest.get("buckets"), dict), "Qualification requires a bucket dictionary")
    graphs = {str(k): (manifest_path.parent / v).resolve() for k, v in manifest["buckets"].items()}
    hashes = {k: sha256_file(p) for k, p in graphs.items()}
    require(set(hashes) == {"768", "1024"} and manifest.get("model_sha256") == hashes, "Frozen graph/manifest bindings differ")
    original_path = ROOT / "reports/qualification-2026-09-29/reference-wsl.jsonl"
    metadata, originals = read_jsonl(original_path)
    typed_suite, typed_ref = typed_probe_reference(rows, metadata, originals, sha256_file(original_path))
    output.mkdir(parents=True)
    (output / "candidate-manifest.json").write_bytes(manifest_path.read_bytes())
    write_json(output / "typed-decisions-full-suite.json", typed_suite)
    typed_ref["suite_file_sha256"] = sha256_file(output / "typed-decisions-full-suite.json")
    write_json(output / "typed-decisions-full-reference.json", typed_ref)
    sources = {entry["name"]: (ROOT / entry["suite"], ROOT / entry["reference"])
               for entry in inventory if "suite" in entry and "reference" in entry}
    sources["typed-decisions-full"] = (output / "typed-decisions-full-suite.json", output / "typed-decisions-full-reference.json")
    sources["legacy"] = (ROOT / "tests/legacy-fidelity-probes.json", args.legacy_reference.resolve())
    entries, common_identity, common_model = [], None, None
    for name in SUITE_NAMES:
        suite_source, ref_source = sources[name]
        suite_path, ref_path = output / f"{name}-suite.json", output / f"{name}-reference.json"
        for source, dest in ((suite_source, suite_path), (ref_source, ref_path)):
            if source.resolve() != dest.resolve():
                dest.write_bytes(source.read_bytes())
        suite, reference = read_json(suite_path), read_json(ref_path)
        validate_reference(suite, reference, suite_path)
        require((len(suite), sum(len(p["questions"]) for p in suite)) == EXPECTED[name], f"Incomplete historical suite {name}")
        identity = upstream_identity(reference)
        common_identity = common_identity or identity
        common_model = common_model or reference["model_files"]
        require(identity == common_identity and reference["model_files"] == common_model, "Frozen references use differing upstream/checkpoint versions")
        entries.append({"name": name, "cases": len(suite), "decisions": EXPECTED[name][1],
                        "suite": suite_path.name, "reference": ref_path.name, "candidate": f"{name}-npu.json",
                        "suite_sha256": json_hash(suite), "suite_file_sha256": sha256_file(suite_path),
                        "reference_sha256": sha256_file(ref_path)})
    require(common_model["model.safetensors"]["sha256"] == manifest["checkpoint_sha256"], "Candidate uses another checkpoint")
    heldout = read_json(ROOT / "reports/evaluation-plans/typed-decisions-heldout.json")
    plan = {"kind": "laya_historical_one_percent_qualification", "format_version": 1,
            "frozen_utc": datetime.now(timezone.utc).isoformat(), "criteria": THRESHOLDS,
            "scope": "Historical regression; not a new independent held-out evaluation. All gates are required separately.",
            "manifest_path": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
            "manifest_evidence": "candidate-manifest.json",
            "graph_paths": {k: str(v) for k, v in graphs.items()}, "graph_sha256_by_bucket": hashes,
            "checkpoint_sha256": manifest["checkpoint_sha256"], "model_files": common_model, "upstream_laya": common_identity,
            "source_typed_reference_sha256": sha256_file(original_path), "source_typed_dataset_sha256": PARQUET_SHA256,
            "historical_inventory_sha256": INVENTORY_SHA256,
            "harness_sha256": {p: sha256_file(ROOT / p) for p in HARNESS}, "suites": entries,
            "subsets": {"typed-historical-heldout": {"suite": "typed-decisions-full", "indices": heldout["selection"]},
                        "typed-development": {"suite": "typed-decisions-full", "indices": list(range(1, 400, 25))},
                        "legacy-accuracy-probes": {"suite": "legacy", "indices": list(range(4))},
                        "legacy-service-requests": {"suite": "legacy", "indices": list(range(4, 7))}}}
    write_json(output / "qualification-plan.json", plan)
    print(f"Frozen {len(entries)} complete suites / {sum(e['decisions'] for e in entries)} decisions in {output}")


def validate_plan(directory):
    plan = read_json(directory / "qualification-plan.json")
    inventory = {entry["name"]: entry for entry in historical_inventory()}
    require(plan.get("kind") == "laya_historical_one_percent_qualification" and plan.get("format_version") == 1, "Unsupported plan")
    require(plan.get("criteria") == THRESHOLDS, "Cannot relax 1% thresholds")
    require(plan.get("historical_inventory_sha256") == INVENTORY_SHA256, "Qualification inventory differs")
    require(plan.get("source_typed_reference_sha256") == inventory["typed-decisions-full"]["reference_sha256"]
            and plan.get("source_typed_dataset_sha256") == PARQUET_SHA256, "Original typed source binding differs")
    require([e["name"] for e in plan["suites"]] == list(SUITE_NAMES), "Historical suite inventory changed")
    frozen_manifest(plan, directory)
    for entry in plan["suites"]:
        require((entry["cases"], entry["decisions"]) == EXPECTED[entry["name"]], "Historical suite coverage changed")
        historical = inventory[entry["name"]]
        if "suite" in historical:
            require(entry["suite_file_sha256"] == historical["suite_sha256"], "Historical suite identity changed")
        if entry["name"] != "typed-decisions-full" and "reference" in historical:
            require(entry["reference_sha256"] == historical["reference_sha256"], "Historical CPU reference changed")
        require(entry["candidate"] == f"{entry['name']}-npu.json", "Candidate report path differs from inventory")
        for key, hash_key in (("suite", "suite_file_sha256"), ("reference", "reference_sha256")):
            path = directory / entry[key]
            require(path.resolve().is_relative_to(directory.resolve()), "Evidence path escapes qualification directory")
            require(sha256_file(path) == entry[hash_key], f"Frozen {key} changed for {entry['name']}")
        if entry["name"] == "typed-decisions-full":
            validate_typed_conversion(read_json(directory / entry["suite"]), read_json(directory / entry["reference"]),
                                      ROOT / historical["reference"], historical["reference_sha256"])
    require(plan["subsets"] == {
        "typed-historical-heldout": {"suite": "typed-decisions-full", "indices": [i for i in range(400) if i % 25 not in (0, 1)]},
        "typed-development": {"suite": "typed-decisions-full", "indices": list(range(1, 400, 25))},
        "legacy-accuracy-probes": {"suite": "legacy", "indices": list(range(4))},
        "legacy-service-requests": {"suite": "legacy", "indices": list(range(4, 7))}}, "Historical subset selection changed")
    return plan


def frozen_manifest(plan, directory):
    require(plan.get("manifest_evidence") == "candidate-manifest.json", "Missing frozen manifest evidence")
    path = directory / plan["manifest_evidence"]
    require(sha256_file(path) == plan["manifest_sha256"], "Frozen manifest evidence hash differs")
    manifest = read_json(path)
    require(manifest.get("model_sha256") == plan["graph_sha256_by_bucket"]
            and manifest.get("checkpoint_sha256") == plan["checkpoint_sha256"], "Frozen manifest graph/checkpoint differs")
    return manifest


def validate_live_assets(plan):
    require(sha256_file(plan["manifest_path"]) == plan["manifest_sha256"], "Frozen candidate manifest changed")
    require({k: sha256_file(p) for k, p in plan["graph_paths"].items()} == plan["graph_sha256_by_bucket"], "Frozen candidate graph changed")
    require({p: sha256_file(ROOT / p) for p in HARNESS} == plan["harness_sha256"], "Frozen inference/audit code changed")


def audit_suite(entry, plan, directory):
    suite_path, ref_path, candidate_path = (directory / entry[k] for k in ("suite", "reference", "candidate"))
    suite, reference, candidate = map(read_json, (suite_path, ref_path, candidate_path))
    validate_reference(suite, reference, suite_path)
    require(json_hash(suite) == entry["suite_sha256"], "Suite semantic hash differs")
    require((candidate.get("kind"), candidate.get("backend"), candidate.get("module"))
            == ("laya_probe_comparison", "npu", "laya_npu"), "Candidate must use strict NPU Laya")
    require(candidate.get("suite_sha256") == entry["suite_sha256"]
            and candidate.get("suite_file_sha256") == entry["suite_file_sha256"]
            and candidate.get("reference_sha256") == entry["reference_sha256"], "Candidate belongs to different frozen inputs/reference")
    require(datetime.fromisoformat(candidate["created_utc"]) >= datetime.fromisoformat(plan["frozen_utc"]), "Inference predates candidate freeze")
    require(candidate.get("model_files") == reference["model_files"] == plan["model_files"], "Checkpoint/tokenizer fingerprints differ")
    require(upstream_identity(candidate) == upstream_identity(reference) == plan["upstream_laya"], "CPU/NPU upstream differs")
    assets, graphs = candidate.get("candidate_assets") or {}, plan["graph_sha256_by_bucket"]
    require({str(k): v.get("sha256") for k, v in assets.get("models", {}).items()} == graphs, "Candidate graph hashes differ")
    require(all(v.get("manifest_verified") is True for v in assets["models"].values())
            and assets.get("checkpoint_manifest_verified") is True, "Missing verified graph/checkpoint binding")
    require(assets.get("manifest_sha256") == plan["manifest_sha256"] and assets.get("checkpoint_sha256") == plan["checkpoint_sha256"]
            and assets.get("manifest", {}).get("model_sha256") == graphs, "Manifest binding differs")
    require(assets["manifest"] == frozen_manifest(plan, directory), "Candidate embedded manifest differs from frozen evidence")
    buckets = sorted(int(k) for k in graphs)
    require(assets.get("supported_buckets") == buckets, "Supported buckets differ")
    require([p["id"] for p in suite] == [r["id"] for r in candidate["records"]], "Candidate complete coverage/order differs")
    all_metrics, per_request, expected_total = [], [], Counter()
    for probe, original, actual in zip(suite, reference["records"], candidate["records"]):
        require(actual["input_sha256"] == original["input_sha256"], "Input identity differs")
        validate_tokens(probe, actual["tokens"], actual["output"])
        require(actual["tokens"] == original["tokens"] and actual.get("tokens_equal") is True
                and actual.get("reported_tokens_equal") is True
                and actual["output"]["usage"] == original["output"]["usage"], "Token/marker inputs differ")
        calls = bucket_calls(actual["tokens"], buckets)
        require(actual.get("npu_execution_verified") is True and actual.get("expected_bucket_calls") == calls
                and actual.get("accelerator_delta") == {"npu_calls": len(probe["questions"]), "cpu_fallbacks": 0, "bucket_calls": calls},
                "Missing actual per-request NPU execution proof")
        expected_total.update(calls)
        metrics = {qid: decision_metrics(q, original["output"]["answers"][qid], actual["output"]["answers"][qid])
                   for qid, q in probe["questions"].items()}
        require(numeric_equal(metrics, actual.get("decisions")), "Stored decision metrics differ from recomputation")
        per_request.append(list(metrics.values()))
        all_metrics.extend(metrics.values())
    stats = candidate.get("accelerator_stats_after_measurement") or {}
    require(stats.get("npu_calls") == len(all_metrics) and stats.get("bucket_calls") == dict(expected_total)
            and stats.get("cpu_fallbacks") == 0, "Aggregate NPU counters differ")
    require(stats.get("backend_fingerprint", {}).get("cpu_ep_fallback") is False, "Backend does not confirm strict execution")
    if "htp_conv_offset_correction" in assets["manifest"]:
        from npu.fidelity_runtime import correction_runtime_requirements, validate_correction_backend
        requirements = correction_runtime_requirements(assets["manifest"], buckets)
        for key in expected_total:
            if int(key) in requirements:
                correction = assets["manifest"]["htp_conv_offset_correction"][key]
                require(correction["output_sha256"] == graphs[key], "Correction graph binding differs")
                require(int(key) in stats.get("correction_backend_verified_buckets", []), "Missing corrected-backend verification")
                validate_correction_backend(requirements[int(key)], stats["backend_fingerprint"], int(key))
    result = gate(all_metrics)
    require(len(all_metrics) == entry["decisions"] and numeric_equal(result["fidelity"], candidate.get("overall")), "Stored aggregate/coverage differs")
    result.update(cases=len(suite), report_sha256=sha256_file(candidate_path), accelerator_stats=stats)
    return result, per_request


def audit(directory, output_path=None):
    plan = validate_plan(directory)
    results, metrics, errors = {}, {}, {}
    for entry in plan["suites"]:
        try:
            results[entry["name"]], metrics[entry["name"]] = audit_suite(entry, plan, directory)
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            errors[entry["name"]] = str(exc)
    for name, subset in plan["subsets"].items():
        if subset["suite"] in metrics:
            results[name] = gate([metric for i in subset["indices"] for metric in metrics[subset["suite"]][i]])
    passed = not errors and all(r["passes_one_percent"] for r in results.values())
    report = {"kind": "laya_one_percent_audit", "created_utc": datetime.now(timezone.utc).isoformat(),
              "plan_sha256": sha256_file(directory / "qualification-plan.json"), "criteria": THRESHOLDS,
              "scope": plan["scope"], "passed": passed, "suites_and_subsets": results, "evidence_errors": errors,
              "notes": ["Each full suite and declared historical subset must pass; pooling cannot hide failures.",
                        "Overlapping typed subsets and MASSIVE translations are not independent examples.",
                        "The 1% limits apply to decision mismatch and mean TV, not every individual probability.",
                        "This historical regression audit makes no new independent held-out generalization claim."]}
    if output_path:
        write_json(output_path, report)
    print(json.dumps({"passed": passed, "errors": errors,
                      "suites": {k: {"decisions": v["fidelity"]["decisions"],
                                      "decision_error_percent": v["fidelity"]["decision_error_percent"],
                                      "mean_tv": v["fidelity"]["total_variation"]["mean"],
                                      "passed": v["passes_one_percent"]} for k, v in results.items()}}, indent=2))
    return report


def next_log_path(directory, name, resume):
    path = directory / f"{name}.log"
    if path.exists():
        require(resume, "Previous attempt log exists; use --resume to preserve it and retry")
        attempt = 2
        while (path := directory / f"{name}.retry-{attempt}.log").exists():
            attempt += 1
    return path


def run(args):
    directory = args.output_dir.resolve()
    plan = validate_plan(directory)
    validate_live_assets(plan)
    env = os.environ.copy()
    env.update(LAYA_NPU_MANIFEST=plan["manifest_path"], LAYA_NPU_CONTEXT_CACHE="1")
    for entry in plan["suites"]:
        dest = directory / entry["candidate"]
        if dest.exists():
            require(args.resume, "Completed output exists; use --resume to audit and reuse it")
            audit_suite(entry, plan, directory)
            continue
        validate_plan(directory)
        validate_live_assets(plan)
        command = [args.python, str(ROOT / "benchmark_probes.py"), "--suite", str(directory / entry["suite"]),
                   "--reference", str(directory / entry["reference"]), "--backend", "npu", "--threads", str(args.threads),
                   "--model-dir", str(args.model_dir.resolve()), "--output", str(dest)]
        with next_log_path(directory, entry["name"], args.resume).open("x", encoding="utf-8") as log:
            completed = subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(f"{entry['name']}: exit {completed.returncode}; evidence {dest}", flush=True)
        # Numerical failures still leave complete reports; every other suite runs.
    return audit(directory, args.audit_output or directory / "one-percent-audit.json")["passed"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare", help="Freeze candidate and all historical original references before inference")
    prepare_parser.add_argument("--dataset-path", type=Path, required=True)
    prepare_parser.add_argument("--legacy-reference", type=Path, required=True)
    prepare_parser.add_argument("--manifest", type=Path, required=True)
    run_parser = sub.add_parser("run", help="Run every frozen suite on strict NPU; then independently audit at 1%")
    run_parser.add_argument("--python", default=sys.executable)
    run_parser.add_argument("--model-dir", type=Path, default=ROOT / "models/multilingual")
    run_parser.add_argument("--threads", type=int, default=4)
    run_parser.add_argument("--resume", action="store_true")
    audit_parser = sub.add_parser("audit", help="Recompute gates with no inference")
    for child in (prepare_parser, run_parser, audit_parser):
        child.add_argument("--output-dir", type=Path, required=True)
    for child in (run_parser, audit_parser):
        child.add_argument("--audit-output", type=Path, help="New audit JSON path; existing evidence is never overwritten")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args)
        return 0
    if args.command == "run":
        require(args.threads > 0, "Thread count must be positive")
        return 0 if run(args) else 1
    return 0 if audit(args.output_dir.resolve(), args.audit_output)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
