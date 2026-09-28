#!/usr/bin/env python3
"""Verify frozen external suites without loading weights or running inference.

From the repository root, with the original Laya/tokenizer dependencies::

    python preflight_external.py --output reports/external-2026-09-29/tokenizer-preflight.json

The report binds the plan, suite bytes/order, tokenizer, upstream source, and
checkpoint identity. Every actual upstream sequence must equal an independently
assembled sequence containing the entire instruction, every entire option, and
the entire state. Marker-count equality alone would miss option-text truncation.
Use a new output path for a later reproduction; existing reports are preserved.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import inspect
import json
from pathlib import Path
import statistics

from benchmark_fidelity import ROOT, json_hash, model_fingerprint, sha256_file


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _lengths(values):
    return {"min": min(values), "max": max(values), "mean": statistics.mean(values)}


def inspect_question(tokenizer, state, question, *, max_len, head_max_len):
    """Reject any input mutation, truncation, duplicate option, or missing marker.

    This deliberately does not copy the upstream budgeting/truncation algorithm.
    It assembles the full format and compares it with the actual upstream builder.
    Importing Agent provides its static conversion only; no Agent is constructed.
    """
    from laya.agent import Agent
    from laya.common import build_sequence, render_options, serialize_state

    _require(question.get("type") == "choice", "External preflight expects choice questions")
    _require(isinstance(question.get("instructions"), str), "Instructions must be text")
    criteria = question.get("criteria")
    _require(isinstance(criteria, (dict, list)) and len(criteria) >= 2,
             "At least two choice criteria are required")
    labels = list(criteria)
    _require(all(isinstance(label, str) and label for label in labels), "Labels must be nonempty text")
    _require(len(labels) == len(set(labels)), "Duplicate option labels would collapse upstream")
    q = Agent._to_internal(question)
    rendered = render_options(q)
    _require(len(rendered) == len(labels), "Question conversion changed option count")
    state_text = serialize_state(state)
    texts = [q["ins"], state_text, *rendered]
    _require(all(tokenizer.mask_token not in text for text in texts),
             "Literal mask tokens would be replaced by upstream preprocessing")

    def encode(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    instruction_ids = encode(q["t"] + " question: " + q["ins"])
    option_ids = [[tokenizer.mask_token_id] + encode(" " + text) for text in rendered]
    state_ids = encode(state_text)
    _require(len(set(tuple(option[1:]) for option in option_ids)) == len(option_ids),
             "Two criteria encode to identical option text")
    expected = [tokenizer.cls_token_id] + instruction_ids + [tokenizer.sep_token_id]
    expected_markers = []
    for option in option_ids:
        expected_markers.append(len(expected))
        expected.extend(option)
    expected += [tokenizer.sep_token_id] + state_ids + [tokenizer.sep_token_id]
    actual, markers = build_sequence(tokenizer, state, q, max_len, head_max_len)
    _require(len(markers) == len(option_ids), "Upstream dropped an option marker")
    _require(markers == sorted(set(markers)), "Option markers are not distinct and increasing")
    _require(all(0 <= marker < len(actual) and actual[marker] == tokenizer.mask_token_id
                 for marker in markers), "Invalid option marker position/token")

    # With all markers present, their spans expose partial option truncation even
    # when the option count is unchanged. The final option ends at the next SEP.
    _require(actual[:markers[0]] == [tokenizer.cls_token_id] + instruction_ids + [tokenizer.sep_token_id],
             "Upstream truncated or changed the instruction")
    try:
        option_end = actual.index(tokenizer.sep_token_id, markers[-1] + 1)
    except ValueError as error:
        raise ValueError("Missing separator after the last option") from error
    for index, (marker, full_option) in enumerate(zip(markers, option_ids)):
        end = markers[index + 1] if index + 1 < len(markers) else option_end
        _require(actual[marker:end] == full_option,
                 f"Upstream truncated or changed option {labels[index]!r}")
    _require(actual[option_end + 1:] == state_ids + [tokenizer.sep_token_id],
             "Upstream truncated or changed the state")
    _require(actual == expected and markers == expected_markers,
             "Sequence differs from the full untruncated format")
    _require(len(actual) <= max_len, "Sequence exceeds the checkpoint limit")
    return {
        "sha256": json_hash({"ids": actual, "markers": markers}),
        "tokens": len(actual), "markers": markers, "state_tokens": len(state_ids),
        "instruction_tokens": len(instruction_ids),
        "option_tokens_including_mask": [len(option) for option in option_ids],
        "option_labels": labels,
    }


def preflight(plan_path, model_dir):
    """Return a complete report, raising before inference on any failed check."""
    plan_path, model_dir = Path(plan_path).resolve(), Path(model_dir).resolve()
    plan = read_json(plan_path)
    _require(plan.get("kind") == "laya_external_fidelity_plan", "Unexpected evaluation plan kind")
    entries = plan.get("suites")
    _require(isinstance(entries, list) and entries, "Plan has no suites")
    _require(len({entry["name"] for entry in entries}) == len(entries), "Duplicate suite names")
    _require(len({entry["suite"] for entry in entries}) == len(entries), "Duplicate suite paths")
    source_lock = ROOT / plan["source_lock"]
    _require(sha256_file(source_lock) == plan["source_lock_sha256"], "Source lock SHA256 differs from plan")
    fingerprints = model_fingerprint(model_dir)  # Streams the weights hash; never loads tensors.
    _require(fingerprints["model.safetensors"]["sha256"] == plan["checkpoint_sha256"],
             "Checkpoint SHA256 differs from plan")
    cfg = read_json(model_dir / "rl_agent_config.json")
    limits = {key: cfg[key] for key in ("max_len", "head_max_len")}
    _require(limits == {"max_len": 1024, "head_max_len": 256}, "Original checkpoint input budgets changed")

    from laya import agent as upstream_agent, common as upstream_common
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"), local_files_only=True)
    report = {
        "kind": "laya_external_tokenizer_preflight", "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan": str(plan_path), "plan_sha256": sha256_file(plan_path),
        "source_lock_sha256": plan["source_lock_sha256"], "model_files": fingerprints,
        "original_input_limits": limits,
        "tool_sha256": sha256_file(__file__),
        "upstream_source_sha256": {
            name: sha256_file(inspect.getfile(module))
            for name, module in (("agent.py", upstream_agent), ("common.py", upstream_common))
        },
        "packages": {name: importlib.metadata.version(name) for name in ("laya", "transformers", "tokenizers")},
        "tokenizer_class": type(tokenizer).__name__,
        "inference_performed": False,
        "checks": ["plan/source-lock/checkpoint/suite hashes", "fixed ordered questions per suite",
                   "complete instructions", "complete option text", "complete state",
                   "unique option labels and encoded text", "unique valid markers",
                   "exact equality to untruncated sequence construction"],
        "suites": [],
        "note": "Tokenization integrity only; no model, accuracy, NPU, or service qualification is performed.",
    }
    for entry in entries:
        suite_path = ROOT / entry["suite"]
        suite = read_json(suite_path)
        _require(sha256_file(suite_path) == entry["suite_file_sha256"],
                 f"{entry['name']}: suite file SHA256 differs from plan")
        _require(json_hash(suite) == entry["suite_sha256"],
                 f"{entry['name']}: suite ordered JSON SHA256 differs from plan")
        _require(isinstance(suite, list) and len(suite) == entry["cases"] and suite,
                 f"{entry['name']}: case count differs from plan")
        seen, records, lengths, state_lengths = set(), [], [], []
        schema = None
        question_summary = {}
        for row in suite:
            case_id = row.get("id")
            _require(isinstance(case_id, str) and case_id and case_id not in seen,
                     f"{entry['name']}: duplicate or invalid case ID {case_id!r}")
            seen.add(case_id)
            _require(isinstance(row.get("state"), str) and row["state"].strip(),
                     f"{case_id}: state must contain only a nonempty text utterance")
            questions = row.get("questions")
            _require(isinstance(questions, dict) and questions, f"{case_id}: missing questions")
            current_schema = json_hash(questions)
            if schema is None:
                schema = current_schema
            _require(current_schema == schema, f"{case_id}: ordered questions/criteria changed within suite")
            details = {}
            for qid, question in questions.items():
                try:
                    detail = inspect_question(tokenizer, row["state"], question, **limits)
                except ValueError as error:
                    raise ValueError(f"{entry['name']}/{case_id}/{qid}: {error}") from error
                question_summary[qid] = {key: detail[key] for key in
                                        ("instruction_tokens", "option_tokens_including_mask", "option_labels")}
                details[qid] = {key: detail[key] for key in ("sha256", "tokens", "markers", "state_tokens")}
                lengths.append(detail["tokens"])
                state_lengths.append(detail["state_tokens"])
            records.append({"id": case_id, "input_sha256": json_hash([row["state"], questions]),
                            "questions": details})
        suite_report = {
            "name": entry["name"], "suite": entry["suite"], "cases": len(records),
            "questions": len(lengths), "suite_sha256": entry["suite_sha256"],
            "suite_file_sha256": entry["suite_file_sha256"], "ordered_question_schema_sha256": schema,
            "sequence_tokens": _lengths(lengths), "state_tokens": _lengths(state_lengths),
            "question_schema": question_summary, "truncated_instructions": 0,
            "truncated_options": 0, "truncated_states": 0,
            "duplicate_option_encodings": 0, "invalid_markers": 0,
            "sequence_records_sha256": json_hash(records), "records": records, "passed": True,
        }
        report["suites"].append(suite_report)
        print(f"{entry['name']}: {len(records)} cases, {min(lengths)}-{max(lengths)} tokens; passed", flush=True)
    report["cases"] = sum(suite["cases"] for suite in report["suites"])
    report["questions"] = sum(suite["questions"] for suite in report["suites"])
    report["passed"] = True
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "reports/evaluation-plans/external-datasets-2026-09-29.json")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/multilingual")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output exists; preserve it and choose a new path")
    try:
        report = preflight(args.plan, args.model_dir)
    except (ValueError, KeyError) as error:
        parser.exit(1, f"Preflight failed: {error}\n")
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"PASS: {report['cases']} cases; no inference; report {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
