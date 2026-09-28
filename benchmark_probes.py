#!/usr/bin/env python3
"""Capture original Laya or compare multilingual and long-input fidelity probes.

This supplementary suite does not replace the pinned held-out benchmark.
Reference capture always imports the unchanged upstream ``laya`` package.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from benchmark_fidelity import (
    ROOT, accelerator_stats, candidate_assets, decision_metrics, json_hash,
    load_agent, model_fingerprint, runtime_info, sha256_file, summarize_decisions,
    token_metadata,
)
from npu.fidelity_runtime import select_bucket


def validate_suite(value):
    if not isinstance(value, list) or not value:
        raise ValueError("Probe suite must be a nonempty list")
    ids = set()
    for probe in value:
        if not isinstance(probe.get("id"), str) or not probe["id"] or probe["id"] in ids:
            raise ValueError("Probe ids must be unique nonempty strings")
        ids.add(probe["id"])
        if not isinstance(probe.get("questions"), dict) or not probe["questions"]:
            raise ValueError("Every probe needs questions")
        if "state" not in probe:
            raise ValueError("Every probe needs state")
        if not 1 <= probe.get("min_sequence_tokens", 1) <= 1024:
            raise ValueError("Invalid minimum sequence length")
    return value


def expected_bucket_calls(tokens, buckets):
    # Upstream Laya collates every question in one predict call. QNNEncoder
    # chooses the bucket once from that collated batch's maximum length.
    bucket = str(select_bucket(buckets, max(item["tokens"] for item in tokens.values())))
    return {bucket: len(tokens)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=ROOT / "tests/fidelity-probes.json")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/multilingual")
    parser.add_argument("--backend", choices=["cpu", "npu"], default="npu")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    args.module = None
    if args.output.exists() or args.threads < 1:
        parser.error("Use a new output path and a positive thread count")
    capturing = args.backend == "cpu" and args.reference is None
    if not capturing and args.reference is None:
        parser.error("NPU comparison requires a pristine CPU --reference")
    suite = validate_suite(json.loads(args.suite.read_text(encoding="utf-8")))
    fingerprint = model_fingerprint(args.model_dir)
    reference = None
    if not capturing:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        if (reference.get("kind") != "laya_probe_reference"
                or reference.get("backend") != "cpu"
                or reference.get("module") != "laya"
                or reference.get("suite_sha256") != json_hash(suite)
                or reference.get("model_files") != fingerprint
                or [p["id"] for p in reference.get("records", [])] != [p["id"] for p in suite]):
            raise ValueError("Reference is incomplete or does not match the suite/checkpoint")
    agent, module = load_agent(args)
    report = {
        "kind": "laya_probe_reference" if capturing else "laya_probe_comparison",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend, "module": module.__name__, "threads": args.threads,
        "suite_sha256": json_hash(suite), "suite_file_sha256": sha256_file(args.suite),
        "model_files": fingerprint, "runtime": runtime_info(),
        "candidate_assets": candidate_assets(agent), "records": [],
        "note": "Supplementary fixed probes; not a held-out dataset qualification. Timings include cold cache and bucket switching.",
    }
    decisions = []
    for index, probe in enumerate(suite):
        tokens = token_metadata(agent, module, probe["state"], probe["questions"], reference=capturing)
        minimum = probe.get("min_sequence_tokens", 1)
        if any(item["tokens"] < minimum for item in tokens.values()):
            raise ValueError(f"Probe {probe['id']} does not exercise its declared minimum length")
        started = time.perf_counter()
        stats_before = accelerator_stats(agent)
        answer = agent.predict(probe["state"], probe["questions"])
        record = {"id": probe["id"], "input_sha256": json_hash([probe["state"], probe["questions"]]),
                  "tokens": tokens, "output": answer, "elapsed_ms": (time.perf_counter() - started) * 1000}
        if args.backend == "npu":
            stats_after = accelerator_stats(agent)
            if stats_before is None or stats_after is None:
                raise ValueError("NPU probes require accelerator counters")
            bucket_counts = stats_after.get("bucket_calls", {})
            previous_counts = stats_before.get("bucket_calls", {})
            delta = {"npu_calls": stats_after["npu_calls"] - stats_before["npu_calls"],
                     "cpu_fallbacks": stats_after["cpu_fallbacks"] - stats_before["cpu_fallbacks"],
                     "bucket_calls": {str(k): v - previous_counts.get(str(k), 0)
                                      for k, v in bucket_counts.items()
                                      if v - previous_counts.get(str(k), 0)}}
            buckets = report["candidate_assets"]["supported_buckets"]
            expected = expected_bucket_calls(tokens, buckets)
            record["accelerator_delta"] = delta
            record["expected_bucket_calls"] = expected
            record["npu_execution_verified"] = (delta["npu_calls"] == len(probe["questions"])
                                                 and delta["cpu_fallbacks"] == 0
                                                 and delta["bucket_calls"] == expected)
        if reference is not None:
            original = reference["records"][index]
            if original["input_sha256"] != record["input_sha256"]:
                raise ValueError("Reference input hash mismatch")
            record["tokens_equal"] = tokens == original["tokens"]
            record["reported_tokens_equal"] = answer["usage"] == original["output"]["usage"]
            record["decisions"] = {
                qid: decision_metrics(question, original["output"]["answers"][qid], answer["answers"][qid])
                for qid, question in probe["questions"].items()
            }
            decisions.extend(record["decisions"].values())
        report["records"].append(record)
        print(f"{probe['id']}: {record['elapsed_ms']:.1f} ms; tokens {[t['tokens'] for t in tokens.values()]}", flush=True)
    report["accelerator_stats_after_measurement"] = accelerator_stats(agent)
    if reference is not None:
        report["reference_sha256"] = sha256_file(args.reference)
        report["overall"] = summarize_decisions(decisions)
        report["passed"] = (
            report["overall"]["decision_error_percent"] <= 5
            and report["overall"]["total_variation"]["mean"] <= .05
            and all(r["tokens_equal"] and r["reported_tokens_equal"] for r in report["records"])
        )
        if args.backend == "npu":
            stats = report["accelerator_stats_after_measurement"] or {}
            report["passed"] = (report["passed"] and stats.get("cpu_fallbacks") == 0
                                and all(r["npu_execution_verified"] for r in report["records"]))
        print(json.dumps(report["overall"], indent=2), flush=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return 0 if report.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
