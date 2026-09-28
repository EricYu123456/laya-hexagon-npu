#!/usr/bin/env python3
"""Compare an accelerated Laya implementation with the unchanged FP32 reference.

Capture once (JSONL is flushed after every case and can be resumed):
  python benchmark_fidelity.py --backend cpu --output reference.jsonl
  python benchmark_fidelity.py --backend cpu --output reference.jsonl --resume
Compare the default NPU runtime or another module exposing load(path, device=...):
  python benchmark_fidelity.py --backend npu --reference reference.jsonl --output fidelity.json
  python benchmark_fidelity.py --module ./laya_candidate.py --reference reference.jsonl \
      --output development.json --indices 0,25,100,125,200,225,300,325
  python benchmark_fidelity.py --reference reference.jsonl --output heldout.json \
      --exclude-indices 0,25,100,125,200,225,300,325
Derive a predeclared held-out report from completed inference, without rerunning it:
  python benchmark_fidelity.py --reference reference.jsonl --output heldout.json \
      --from-candidate-records full.cases.jsonl --source-report full.json \
      --selection-plan reports/evaluation-plans/typed-decisions-heldout.json

The full pinned public test split is the default. Subset reports cannot certify the
full split. Do not use evaluation cases to fit calibration ranges or select models.
Gold labels are deliberately excluded: this benchmark measures agreement with Laya,
not the correctness of Laya itself. Probability errors use API outputs rounded to
four decimals, normalized to remove their tiny rounding-induced sum error.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
DATASET = "LocalLLaMA/typed-decisions"
REVISION = "c76749ec58bd8c3d2ea706b31c333a9059c38f90"
PARQUET = "all/test-00000-of-00001.parquet"
PARQUET_SHA256 = "4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c"
FORMAT_VERSION = 1


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    # Preserve mapping order: the order of choice criteria determines marker order.
    data = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def dump_line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def finite_number(value, label, low=None, high=None):
    number = float(value)
    if not math.isfinite(number) or (low is not None and number < low) or (high is not None and number > high):
        raise ValueError(f"Invalid {label}: {value!r}")
    return number


def distribution(answer, labels, qtype):
    if qtype == "noul":
        p = finite_number(answer["noul"], "noul", 0, 1)
        return [1 - p, p]
    if set(answer["probabilities"]) != set(labels):
        raise ValueError("Probability labels differ from question criteria")
    values = [finite_number(answer["probabilities"][label], "probability", 0, 1) for label in labels]
    total = sum(values)
    if total <= 0 or abs(total - 1.0) > max(0.001, len(labels) * 0.000051):
        raise ValueError(f"Probability sum is {total}, expected 1")
    return [value / total for value in values]


def decision_metrics(question, reference, candidate):
    """Return independent decision, distribution, score and metadata errors."""
    qtype = question["type"]
    if reference["type"] != qtype or candidate["type"] != qtype:
        raise ValueError("Answer type does not match question")
    criteria = question.get("criteria")
    labels = list(criteria) if qtype == "choice" else [str(i) for i in range(len(criteria))] if qtype == "score" else ["false", "true"]
    if not labels:
        raise ValueError("Question must have at least one option")
    p = distribution(reference, labels, qtype)
    q = distribution(candidate, labels, qtype)
    differences = [abs(a - b) for a, b in zip(p, q)]
    ref_top = max(range(len(p)), key=p.__getitem__)
    cand_top = max(range(len(q)), key=q.__getitem__)
    result = {
        "total_variation": 0.5 * sum(differences),
        "probability_mae": statistics.mean(differences),
        "max_probability_error": max(differences),
        "argmax_mismatch": int(ref_top != cand_top),
        "confidence_abs_error": abs(finite_number(reference["confidence"], "confidence", 0, 1) - finite_number(candidate["confidence"], "confidence", 0, 1)),
        "action_probability_abs_error": abs(finite_number(reference["action"]["act_probability"], "action probability", 0, 1) - finite_number(candidate["action"]["act_probability"], "action probability", 0, 1)),
    }
    if qtype == "choice":
        if reference["choice"] not in labels or candidate["choice"] not in labels:
            raise ValueError("Choice answer not present in criteria")
        # Published choice can differ from the argmax of rounded probabilities at ties.
        result["decision_mismatch"] = int(reference["choice"] != candidate["choice"])
    elif qtype == "noul":
        result["decision_mismatch"] = int((reference["noul"] > 0.5) != (candidate["noul"] > 0.5))
        result["noul_abs_error"] = differences[1]
    else:
        ref_score = finite_number(reference["score"], "score", 0, len(labels) - 1)
        cand_score = finite_number(candidate["score"], "score", 0, len(labels) - 1)
        result["score_abs_error"] = abs(ref_score - cand_score)
        result["score_normalized_abs_error"] = abs(ref_score - cand_score) / max(1, len(labels) - 1)
        # Retain the old benchmark's score decision convention for comparability;
        # argmax_mismatch above separately measures the categorical top option.
        result["decision_mismatch"] = int(round(ref_score) != round(cand_score))
    return result


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize_values(values):
    return {"count": len(values), "mean": statistics.mean(values), "p50": percentile(values, 0.5), "p95": percentile(values, 0.95), "max": max(values)} if values else None


def summarize_decisions(decisions):
    if not decisions:
        return {"decisions": 0}
    errors = sum(item["decision_mismatch"] for item in decisions)
    summary = {
        "decisions": len(decisions), "decision_mismatches": errors,
        "decision_error_percent": 100 * errors / len(decisions),
        "argmax_mismatches": sum(item["argmax_mismatch"] for item in decisions),
        "argmax_error_percent": 100 * statistics.mean(item["argmax_mismatch"] for item in decisions),
    }
    for key in ("total_variation", "probability_mae", "max_probability_error", "noul_abs_error", "score_abs_error", "score_normalized_abs_error", "confidence_abs_error", "action_probability_abs_error"):
        summary[key] = summarize_values([item[key] for item in decisions if key in item])
    return summary


def read_records(path, kind):
    with Path(path).open(encoding="utf-8") as stream:
        lines = [json.loads(line) for line in stream if line.strip()]
    if not lines or lines[0].get("kind") != "metadata" or lines[0].get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported or empty {kind} records: {path}")
    rows = {}
    for entry in lines[1:]:
        if entry.get("kind") != kind or type(entry.get("index")) is not int or entry["index"] in rows:
            raise ValueError(f"{kind} records contain invalid or duplicate rows")
        rows[entry["index"]] = entry
    return lines[0], rows


def read_jsonl(path):
    return read_records(path, "reference")


def load_dataset(path=None):
    import pyarrow.parquet as pq
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(DATASET, filename=PARQUET, repo_type="dataset", revision=REVISION)
    if sha256_file(path) != PARQUET_SHA256:
        raise ValueError("Dataset checksum differs from the pinned public benchmark")
    return pq.read_table(path).to_pylist()


def parse_row(row, index):
    decode = lambda value: json.loads(value) if isinstance(value, str) else value
    state, questions = decode(row["state"]), decode(row["questions"])
    return {"index": index, "id": row["id"], "workflow": row["workflow"], "input_sha256": json_hash([state, questions])}, state, questions


def select_indices(total, args):
    if args.indices:
        indices = [int(part.strip()) for part in args.indices.split(",")]
        if args.offset or args.limit is not None:
            raise ValueError("--indices cannot be combined with --offset or --limit")
    else:
        if args.offset < 0 or (args.limit is not None and args.limit <= 0):
            raise ValueError("--offset must be nonnegative and --limit must be positive")
        indices = list(range(args.offset, min(total, args.offset + args.limit) if args.limit is not None else total))
    if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= total for i in indices):
        raise ValueError("Selection must contain distinct valid dataset indices")
    excluded = excluded_indices(total, args)
    indices = [index for index in indices if index not in excluded]
    if not indices:
        raise ValueError("Exclusions leave no selected cases")
    return indices


def excluded_indices(total, args):
    text = getattr(args, "exclude_indices", None)
    excluded = [int(part.strip()) for part in text.split(",")] if text else []
    if len(excluded) != len(set(excluded)) or any(i < 0 or i >= total for i in excluded):
        raise ValueError("Exclusions must contain distinct valid dataset indices")
    return set(excluded)


def model_fingerprint(model_dir):
    names = ["model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]
    fingerprints = {}
    for name in names:
        if name.endswith(".json"):
            value = json.loads((model_dir / name).read_text(encoding="utf-8-sig"))
            canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            fingerprints[name] = {"semantic_json_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}
        else:
            fingerprints[name] = {"sha256": sha256_file(model_dir / name)}
    return fingerprints


def runtime_info():
    packages = {}
    for name in ("laya", "torch", "transformers", "onnxruntime", "onnxruntime-qnn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    spec = importlib.util.find_spec("laya")
    sources = {}
    if spec and spec.submodule_search_locations:
        package_dir = Path(next(iter(spec.submodule_search_locations)))
        sources = {name: sha256_file(package_dir / name) for name in ("__init__.py", "agent.py", "common.py")}
    return {"python": platform.python_version(), "platform": platform.platform(), "hostname": platform.node(), "packages": packages, "laya_source_files": sources}


def load_agent(args):
    import torch
    torch.set_num_threads(args.threads)
    module_name = args.module or ("laya" if args.backend == "cpu" else "laya_npu")
    if module_name.endswith(".py") or Path(module_name).is_file():
        spec = importlib.util.spec_from_file_location("fidelity_candidate", module_name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    agent = module.load(str(args.model_dir), device=args.backend)
    return agent, module


def token_metadata(agent, module, state, questions, reference=False):
    """Fingerprint preprocessing, separately from self-reported token usage.

    Custom runtimes can expose fidelity_sequences(state, questions), returning
    {qid: {"ids": [...], "markers": [...]}} to describe their actual inputs.
    Otherwise use original Laya's builder and the runtime's published limits.
    """
    if hasattr(agent, "fidelity_sequences"):
        sequences = agent.fidelity_sequences(state, questions)
    else:
        from laya.common import build_sequence
        cfg = agent.cfg
        max_len = cfg.get("max_len", 512) if reference else getattr(agent, "max_len", getattr(module, "MAX_SEQ_LEN", cfg.get("max_len", 512)))
        head_len = cfg.get("head_max_len", 192) if reference else getattr(agent, "head_max_len", getattr(module, "HEAD_MAX_LEN", cfg.get("head_max_len", 192)))
        sequences = {}
        for qid, question in questions.items():
            ids, markers = build_sequence(agent.tok, state, agent._to_internal(question), max_len, head_len)
            sequences[qid] = {"ids": ids, "markers": markers}
    if set(sequences) != set(questions):
        raise ValueError("Preprocessing metadata has different question ids")
    return {qid: {"sha256": json_hash(value), "tokens": len(value["ids"]), "markers": value["markers"]} for qid, value in sequences.items()}


def candidate_assets(agent):
    """Record the actual manifest and model bytes, not only checkpoint weights."""
    if hasattr(agent, "fidelity_metadata"):
        return agent.fidelity_metadata()
    if not hasattr(agent, "manifest_path"):
        return None
    path = Path(agent.manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    buckets = manifest["buckets"]
    entries = buckets.items() if isinstance(buckets, dict) else ((entry["sequence_length"], entry["path"]) for entry in buckets)
    assets = {}
    for length, filename in entries:
        model = Path(filename)
        if not model.is_absolute():
            model = path.parent / model
        assets[str(length)] = {"path": str(model), "sha256": sha256_file(model), "bytes": model.stat().st_size}
    return {"manifest_path": str(path), "manifest_sha256": sha256_file(path), "manifest": manifest, "models": assets}


def accelerator_stats(agent):
    # Snapshot mutable counters so later predictions cannot rewrite the baseline.
    return json.loads(json.dumps(agent.npu_stats)) if hasattr(agent, "npu_stats") else None


def check_reference_metadata(metadata, fingerprint):
    if metadata.get("dataset") != {"id": DATASET, "revision": REVISION, "file": PARQUET, "sha256": PARQUET_SHA256}:
        raise ValueError("Reference cache was produced with a different dataset")
    if metadata.get("model_files") != fingerprint:
        raise ValueError("Reference cache does not match these checkpoint/tokenizer files")
    if metadata.get("backend") != "cpu" or metadata.get("module") != "laya":
        raise ValueError("Reference must come from unchanged laya with --backend cpu")


def validate_reference_row(reference, identity):
    for key in ("index", "id", "workflow", "input_sha256"):
        if reference.get(key) != identity[key]:
            raise ValueError(f"Reference row {identity['index']} has mismatching {key}")


def aggregate_report(metadata, reference_metadata, reference_hash, reference_cases,
                     case_records, records_path, final_stats=None, declared_heldout=None):
    """Aggregate measured rows; cumulative counters are supplied only by a real run."""
    all_metrics = [metric for record in case_records for metric in record["metrics"].values()]
    overall = summarize_decisions(all_metrics)
    input_equal = all(record["token_sequences_equal"] for record in case_records)
    passed = bool(all_metrics) and overall["decision_error_percent"] <= 5.0 and overall["total_variation"]["mean"] <= 0.05 and input_equal
    indices = metadata["selection"]
    total = metadata["total_dataset_cases"]
    excluded = set(metadata.get("excluded_calibration_or_development_indices", []))
    complete = len(indices) == total and set(indices) == set(range(total))
    heldout = bool(excluded) and set(indices) == set(range(total)) - excluded
    if declared_heldout is not None:
        heldout = heldout and declared_heldout
    comparable_latency = all(reference_metadata["runtime"].get(key) == metadata["runtime"].get(key) and metadata["runtime"].get(key) is not None for key in ("hostname", "platform")) and reference_metadata.get("threads") == metadata.get("threads")
    return {
        "metadata": metadata, "reference_metadata": reference_metadata,
        "reference_cache_sha256": reference_hash, "reference_cache_cases": reference_cases,
        "accelerator_stats_after_measurement": final_stats,
        "cases": len(case_records), "complete_public_split": complete,
        "evaluation_scope": "full_public_split" if complete else "heldout_selection" if heldout else "development_subset",
        "complete_declared_heldout_selection": heldout,
        "criteria": {"decision_error_percent_max": 5.0, "mean_total_variation_max": 0.05, "identical_preprocessed_inputs_required": True},
        "selected_cases_pass": passed, "full_public_split_pass": passed and complete,
        "heldout_selection_pass": passed and heldout,
        "overall": overall,
        "by_type": {value: summarize_decisions([item for item in all_metrics if item["type"] == value]) for value in sorted({item["type"] for item in all_metrics})},
        "by_workflow": {value: summarize_decisions([item for item in all_metrics if item["workflow"] == value]) for value in sorted({item["workflow"] for item in all_metrics})},
        "token_sequences_equal_cases": sum(record["token_sequences_equal"] for record in case_records),
        "reported_token_counts_equal_cases": sum(record["reported_token_counts_equal"] for record in case_records),
        "latency_case_ms": {"reference": summarize_values([record["cpu_ms"] for record in case_records]), "candidate": summarize_values([record["candidate_ms"] for record in case_records])},
        "latency_same_host_and_threads": comparable_latency,
        "notes": ["Fidelity is relative to unchanged FP32 Laya, not gold-label accuracy.", "Decision mismatch uses choice labels, noul > 0.5, and rounded expected scores; argmax error is reported separately.", "Probability errors are fractions in [0,1], not relative percent errors; distributions are normalized after API rounding.", "Latency includes tokenization and heads, excludes model loading and warmup. Cross-host timings are not comparable and cannot establish hardware speedup.", "Exclusions are declared calibration/development cases; define them before evaluating candidates, never after observing failures.", "A test-set result does not guarantee fidelity for arbitrary inputs; evaluation covers only the recorded selection."],
        "records_file": str(records_path),
    }


def valid_index_list(values, total, label, allow_empty=False):
    if not isinstance(values, list) or any(type(i) is not int or i < 0 or i >= total for i in values) or len(set(values)) != len(values) or (not values and not allow_empty):
        raise ValueError(f"{label} must contain distinct valid dataset indices")
    return values


def parse_utc(value):
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("Evidence timestamps must include a timezone")
    return timestamp.astimezone(timezone.utc)


def verify_selection_plan(path, source_metadata, total, repository=ROOT):
    """Verify the historical Git document, without backdating a plan serialized later.

    The documented EXCLUDED shell assignment is the machine-readable declaration.
    A local Git timestamp is repository evidence, not an independent time authority.
    """
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan.get("kind") != "heldout_selection_plan" or plan.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported held-out selection plan")
    if plan.get("dataset") != source_metadata["dataset"] or plan.get("total_dataset_cases") != total:
        raise ValueError("Selection plan uses a different dataset")
    indices = valid_index_list(plan.get("selection"), total, "Plan selection")
    excluded = valid_index_list(plan.get("excluded_calibration_or_development_indices"), total, "Plan exclusions")
    if set(indices) != set(range(total)) - set(excluded):
        raise ValueError("Plan selection is not the complete complement of its exclusions")
    declaration = plan.get("declaration", {})
    commit, document = declaration.get("commit", ""), declaration.get("path", "")
    if declaration.get("kind") != "git_documentation" or not re.fullmatch(r"[0-9a-f]{40}", commit) or not document or document.startswith(("/", "-")) or ":" in document:
        raise ValueError("Plan requires a full Git commit and documentation path")
    def git(*arguments):
        return subprocess.run(["git", "-C", str(repository), *arguments], check=True, capture_output=True).stdout
    try:
        contents = git("show", f"{commit}:{document}")
        committed_utc = git("show", "-s", "--format=%cI", commit).decode().strip()
    except subprocess.CalledProcessError as error:
        raise ValueError("Cannot verify the selection declaration in local Git history") from error
    digest = hashlib.sha256(contents).hexdigest()
    if digest != declaration.get("document_sha256"):
        raise ValueError("Committed selection document hash differs from the plan")
    assignments = re.findall(r"^EXCLUDED=([0-9,]+)\s*$", contents.decode("utf-8"), flags=re.MULTILINE)
    if len(assignments) != 1 or [int(i) for i in assignments[0].split(",")] != excluded:
        raise ValueError("Plan exclusions differ from the committed EXCLUDED declaration")
    if parse_utc(committed_utc) >= parse_utc(source_metadata["created_utc"]):
        raise ValueError("Held-out declaration does not predate candidate inference")
    if declaration.get("committed_utc") != committed_utc:
        raise ValueError("Plan commit timestamp differs from Git history")
    return plan, {
        "path": str(path), "sha256": sha256_file(path),
        "plan_created_utc": plan.get("created_utc"),
        "declaration": declaration, "declaration_verified_against_git": True,
        "declaration_predates_source_run": True,
        "timestamp_basis": "Local Git committer timestamp; plan materialization time is not treated as predeclaration time.",
    }


def recompute_comparison(record, baseline, dataset_row, index):
    identity, _, questions = parse_row(dataset_row, index)
    validate_reference_row(record, identity)
    validate_reference_row(baseline, identity)
    if record.get("cpu") != baseline["cpu"] or record.get("reference_tokens") != baseline["tokens"] or record.get("cpu_ms") != baseline["cpu_ms"]:
        raise ValueError(f"Comparison row {index} does not contain the supplied reference")
    answer, tokens = record["candidate"], record["candidate_tokens"]
    if set(answer["answers"]) != set(questions) or set(baseline["cpu"]["answers"]) != set(questions) or set(tokens) != set(questions) or set(baseline["tokens"]) != set(questions):
        raise ValueError(f"Answer/token question ids differ at case {index}")
    for label, measured, token_data in (("Candidate", answer, tokens), ("Reference", baseline["cpu"], baseline["tokens"])):
        if any(type(info.get("tokens")) is not int or info["tokens"] <= 0 or not isinstance(info.get("sha256"), str) or not isinstance(info.get("markers"), list) for info in token_data.values()):
            raise ValueError(f"{label} preprocessing evidence is invalid at case {index}")
        if measured["usage"]["input_tokens"] != sum(info["tokens"] for info in token_data.values()):
            raise ValueError(f"{label} token count disagrees with preprocessing evidence at case {index}")
    metrics = {qid: {"qid": qid, "type": question["type"], "workflow": identity["workflow"], **decision_metrics(question, baseline["cpu"]["answers"][qid], answer["answers"][qid])} for qid, question in questions.items()}
    # Cached metrics and equality flags are deliberately discarded and recomputed.
    return {**record, "metrics": metrics, "token_sequences_equal": tokens == baseline["tokens"],
            "reported_token_counts_equal": answer["usage"]["input_tokens"] == baseline["cpu"]["usage"]["input_tokens"],
            "cpu_ms": finite_number(record["cpu_ms"], "reference latency", 0),
            "candidate_ms": finite_number(record["candidate_ms"], "candidate latency", 0)}


def derive_offline_report(args):
    """Validate completed evidence and select rows without importing a model runtime."""
    records_path = args.output.with_suffix(".cases.jsonl")
    sources = [args.reference, args.from_candidate_records, args.source_report, args.selection_plan]
    for destination in (args.output, records_path):
        if destination.exists() or any(source is not None and destination.resolve() == source.resolve() for source in sources):
            raise ValueError(f"Offline report cannot overwrite evidence: {destination}")
    if records_path.resolve() == args.output.resolve():
        raise ValueError("Report and candidate records must have distinct paths")
    rows = load_dataset(args.dataset_path)
    source_metadata, source_records = read_records(args.from_candidate_records, "comparison")
    reference_metadata, cached = read_jsonl(args.reference)
    check_reference_metadata(reference_metadata, source_metadata.get("model_files"))
    if source_metadata.get("dataset") != reference_metadata["dataset"] or source_metadata.get("total_dataset_cases") != len(rows):
        raise ValueError("Candidate records use a different dataset")
    if source_metadata.get("execution") == "offline_subset_no_inference":
        raise ValueError("Derive from original inference records, not an already derived subset")
    source_indices = valid_index_list(source_metadata.get("selection"), len(rows), "Original run selection")
    if set(source_records) != set(source_indices):
        raise ValueError("Original candidate inference is incomplete or contains unexpected rows")
    if any(index not in cached for index in source_indices):
        raise ValueError("Reference cache does not cover the original run")
    recomputed = {index: recompute_comparison(source_records[index], cached[index], rows[index], index) for index in source_indices}
    plan_evidence = None
    if args.selection_plan:
        if args.indices or args.exclude_indices or args.offset or args.limit is not None:
            raise ValueError("--selection-plan cannot be combined with another row selection")
        plan, plan_evidence = verify_selection_plan(args.selection_plan, source_metadata, len(rows))
        indices = plan["selection"]
        excluded = plan["excluded_calibration_or_development_indices"]
    else:
        indices = select_indices(len(rows), args)
        excluded = sorted(excluded_indices(len(rows), args))
    if any(index not in recomputed for index in indices):
        raise ValueError("Candidate records do not cover every selected case")
    reference_hash = sha256_file(args.reference)
    source_report = None
    source_report_evidence = None
    if args.source_report:
        source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
        if source_report.get("metadata") != source_metadata or source_report.get("reference_cache_sha256") != reference_hash or source_report.get("reference_metadata") != reference_metadata or source_report.get("reference_cache_cases") != len(cached):
            raise ValueError("Source report does not belong to these original records/reference")
        rebuilt = aggregate_report(source_metadata, reference_metadata, reference_hash, len(cached), list(recomputed.values()), args.from_candidate_records)
        for key in ("cases", "overall", "by_type", "by_workflow", "token_sequences_equal_cases", "reported_token_counts_equal_cases"):
            if source_report.get(key) != rebuilt[key]:
                raise ValueError(f"Source report {key} differs from recomputed original records")
        source_report_evidence = {"path": str(args.source_report), "sha256": sha256_file(args.source_report)}
    original_run = {
        "metadata": source_metadata, "cases": len(source_records),
        "records": {"path": str(args.from_candidate_records), "sha256": sha256_file(args.from_candidate_records)},
        "report": source_report_evidence,
        "accelerator_stats_after_measurement": source_report.get("accelerator_stats_after_measurement") if source_report else None,
        "counter_scope": "Entire original inference run, including its warmup; never attributed to the derived selection.",
    }
    metadata = {**source_metadata, "created_utc": datetime.now(timezone.utc).isoformat(),
                "inference_created_utc": source_metadata["created_utc"], "execution": "offline_subset_no_inference",
                "selection": indices, "excluded_calibration_or_development_indices": excluded,
                "warmup_cases": None, "accelerator_stats_after_warmup": None}
    selected = [recomputed[index] for index in indices]
    report = aggregate_report(metadata, reference_metadata, reference_hash, len(cached), selected, records_path, declared_heldout=plan_evidence is not None)
    if not report["complete_public_split"] and plan_evidence is None:
        report["evaluation_scope"] = "posthoc_subset"
    report["original_run"] = original_run
    report["derivation"] = {"kind": "offline_subset_no_inference", "metrics_recomputed_from_answers": True,
                            "selection_plan": plan_evidence, "selected_dataset_indices": indices,
                            "excluded_dataset_indices": excluded, "python": platform.python_version(),
                            "hostname": platform.node(), "benchmark_source_sha256": sha256_file(__file__)}
    report["notes"].extend([
        "No inference was performed during this derivation. Runtime/model provenance and timings describe the original recorded inference.",
        "Aggregate accelerator counters apply to original_run only. There are no measured subset-specific accelerator counters.",
        "An offline subset without a verified declaration predating the original run cannot certify a held-out pass.",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("x", encoding="utf-8") as stream:
        dump_line(stream, {**metadata, "original_run": original_run, "derivation": report["derivation"]})
        for record in selected:
            dump_line(stream, record)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("cases", "evaluation_scope", "selected_cases_pass", "heldout_selection_pass", "overall")}, indent=2))
    print(f"Derived report: {args.output}; selected evidence: {records_path}")
    return 0 if report["selected_cases_pass"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("cpu", "npu"), default="npu")
    parser.add_argument("--module", help="Candidate import name or .py file exposing load(path, device=...)")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/multilingual")
    parser.add_argument("--dataset-path", type=Path, help="Use local pinned parquet without contacting the Hub")
    parser.add_argument("--reference", type=Path, help="Previously captured FP32 JSONL")
    parser.add_argument("--from-candidate-records", type=Path, help="Derive metrics from completed original comparison JSONL; performs no inference")
    parser.add_argument("--source-report", type=Path, help="Original aggregate report, retaining its cumulative accelerator counters")
    parser.add_argument("--selection-plan", type=Path, help="Held-out plan backed by a verified Git declaration predating original inference")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", help="Comma-separated zero-based row indices for development")
    parser.add_argument("--exclude-indices", help="Comma-separated calibration/development indices fixed before held-out evaluation")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Resume a partial CPU reference capture")
    args = parser.parse_args(argv)
    if args.from_candidate_records:
        if args.reference is None or args.resume:
            parser.error("Offline derivation requires --reference and does not support --resume")
        return derive_offline_report(args)
    if args.source_report or args.selection_plan:
        parser.error("--source-report and --selection-plan require --from-candidate-records")
    capturing = args.backend == "cpu" and args.reference is None
    if args.threads <= 0 or args.warmup < 0:
        parser.error("--threads must be positive; --warmup must be nonnegative")
    if capturing and args.module not in (None, "laya"):
        parser.error("Reference capture must use unchanged laya; use --reference for candidate comparisons")
    if not capturing and args.reference is None:
        parser.error("Candidate comparison requires --reference")
    if args.resume and not capturing:
        parser.error("--resume is only valid for reference capture")
    if args.reference and args.output.resolve() == args.reference.resolve():
        parser.error("--output cannot overwrite --reference")
    if args.output.exists() and not args.resume:
        parser.error("Output already exists; choose a new path or --resume the CPU capture")

    rows = load_dataset(args.dataset_path)
    indices = select_indices(len(rows), args)
    excluded = excluded_indices(len(rows), args)
    fingerprint = model_fingerprint(args.model_dir)
    metadata = {
        "kind": "metadata", "format_version": FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"id": DATASET, "revision": REVISION, "file": PARQUET, "sha256": PARQUET_SHA256},
        "model_files": fingerprint, "backend": args.backend, "module": args.module or ("laya" if args.backend == "cpu" else "laya_npu"),
        "runtime": runtime_info(), "threads": args.threads, "warmup_cases": args.warmup,
        "selection": indices, "total_dataset_cases": len(rows),
        "excluded_calibration_or_development_indices": sorted(excluded),
    }
    cached = {}
    reference_metadata = None
    if capturing and args.output.exists():
        reference_metadata, cached = read_jsonl(args.output)
        check_reference_metadata(reference_metadata, fingerprint)
        if reference_metadata.get("threads") != args.threads:
            raise ValueError("Cannot mix thread settings while resuming latency measurements")
        for key in ("packages", "laya_source_files"):
            if reference_metadata["runtime"].get(key) != metadata["runtime"].get(key):
                raise ValueError(f"Cannot resume a reference cache after changing runtime {key}")
    elif not capturing:
        reference_metadata, cached = read_jsonl(args.reference)
        check_reference_metadata(reference_metadata, fingerprint)
        missing = [index for index in indices if index not in cached]
        if missing:
            raise ValueError(f"Reference cache missing {len(missing)} selected cases, starting with {missing[:5]}")
    for index in indices:
        if index in cached:
            validate_reference_row(cached[index], parse_row(rows[index], index)[0])
    pending = [index for index in indices if index not in cached] if capturing else indices
    if not pending:
        print(f"All {len(indices)} selected reference cases already captured.")
        return 0

    print(f"Loading {metadata['module']}; {len(pending)} cases; {args.threads} CPU threads", flush=True)
    agent, module = load_agent(args)
    metadata["module_source_sha256"] = sha256_file(module.__file__)
    metadata["candidate_assets"] = candidate_assets(agent)
    _, warm_state, warm_questions = parse_row(rows[pending[0]], pending[0])
    for _ in range(args.warmup):
        agent.predict(warm_state, warm_questions)
    metadata["accelerator_stats_after_warmup"] = accelerator_stats(agent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records_path = args.output if capturing else args.output.with_suffix(".cases.jsonl")
    if not capturing and records_path.resolve() in (args.output.resolve(), args.reference.resolve()):
        raise ValueError("Report, candidate records and reference must have distinct paths")
    if not capturing and records_path.exists():
        raise ValueError(f"Candidate records already exist: {records_path}")
    all_metrics, case_records = [], []
    mode = "a" if capturing and args.output.exists() else "x"
    with records_path.open(mode, encoding="utf-8") as stream:
        if mode != "a":
            dump_line(stream, metadata)
        for position, index in enumerate(pending, 1):
            identity, state, questions = parse_row(rows[index], index)
            tokens = token_metadata(agent, module, state, questions, reference=capturing)
            started = time.perf_counter()
            answer = agent.predict(state, questions)
            elapsed = (time.perf_counter() - started) * 1000
            if set(answer["answers"]) != set(questions):
                raise ValueError(f"Answer question ids differ at case {index}")
            if answer["usage"]["input_tokens"] != sum(info["tokens"] for info in tokens.values()):
                raise ValueError(f"Runtime token count disagrees with preprocessing metadata at case {index}")
            if capturing:
                record = {"kind": "reference", **identity, "cpu": answer, "cpu_ms": elapsed, "tokens": tokens}
                # Validate captured distributions as well as candidate distributions.
                for qid, question in questions.items():
                    decision_metrics(question, answer["answers"][qid], answer["answers"][qid])
            else:
                baseline = cached[index]
                metrics = {}
                for qid, question in questions.items():
                    metrics[qid] = {"qid": qid, "type": question["type"], "workflow": identity["workflow"], **decision_metrics(question, baseline["cpu"]["answers"][qid], answer["answers"][qid])}
                    all_metrics.append(metrics[qid])
                record = {"kind": "comparison", **identity, "cpu": baseline["cpu"], "cpu_ms": baseline["cpu_ms"], "candidate": answer, "candidate_ms": elapsed, "reference_tokens": baseline["tokens"], "candidate_tokens": tokens, "token_sequences_equal": tokens == baseline["tokens"], "reported_token_counts_equal": answer["usage"]["input_tokens"] == baseline["cpu"]["usage"]["input_tokens"], "metrics": metrics}
                case_records.append(record)
            dump_line(stream, record)
            print(f"[{position}/{len(pending)}] row={index} {identity['workflow']} {elapsed:.1f} ms", flush=True)

    if capturing:
        print(f"Saved FP32 reference: {args.output}; selected cases complete: {len(indices)}", flush=True)
        return 0
    report = aggregate_report(metadata, reference_metadata, sha256_file(args.reference), len(cached),
                              case_records, records_path, final_stats=accelerator_stats(agent))
    passed = report["selected_cases_pass"]
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("cases", "complete_public_split", "selected_cases_pass", "full_public_split_pass", "overall")}, indent=2))
    print(f"Report: {args.output}; per-case evidence: {records_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
