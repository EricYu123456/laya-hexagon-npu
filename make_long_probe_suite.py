#!/usr/bin/env python3
"""Materialize predeclared full-length probes from the pinned public dataset.

Only the tokenizer and upstream question conversion are loaded. The generated
suite is consumed by benchmark_probes.py; no model answers are inspected here.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from benchmark_fidelity import DATASET, REVISION, PARQUET_SHA256, ROOT, load_dataset, json_hash, sha256_file


def make_suite(rows, indices, tokenizer, convert_question, max_len, head_max_len):
    if (not indices or len(indices) != len(set(indices))
            or any(type(i) is not int or not 0 <= i < len(rows) for i in indices)):
        raise ValueError("Indices must be distinct valid rows")
    from laya.common import build_sequence
    suite = []
    for index in indices:
        row = rows[index]
        state, questions = json.loads(row["state"]), json.loads(row["questions"])
        if not isinstance(state, (str, dict, list)) or not state:
            raise ValueError("Long probes require a nonempty supported state")
        repeated = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        if not repeated.strip() or not questions:
            raise ValueError("Long probes require nonempty states and questions")
        for _ in range(12):
            lengths = [len(build_sequence(tokenizer, repeated, convert_question(question),
                                           max_len, head_max_len)[0]) for question in questions.values()]
            if all(length == max_len for length in lengths):
                suite.append({"id": f"heldout-long-{index}", "source_index": index,
                              "workflow": row["workflow"], "state": repeated,
                              "questions": questions, "min_sequence_tokens": max_len})
                break
            repeated += "\n\n" + repeated
        else:
            raise ValueError(f"Row {index} did not fill the original input budget")
    return suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "reports/evaluation-plans/long-input-heldout.json")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/multilingual")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".metadata.json").exists():
        parser.error("Output or metadata already exists")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("dataset") != {"id": DATASET, "revision": REVISION, "sha256": PARQUET_SHA256}:
        raise ValueError("Plan does not identify the pinned dataset")
    if set(plan["indices"]) & set(plan["excluded_calibration_and_development_indices"]):
        raise ValueError("Long validation overlaps calibration/development rows")
    cfg = json.loads((args.model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    limits = {key: cfg[key] for key in ("max_len", "head_max_len")}
    if limits != plan["original_input_limits"]:
        raise ValueError("Plan and original checkpoint input budgets differ")
    from laya.agent import Agent
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir / "tokenizer"), local_files_only=True)
    suite = make_suite(load_dataset(args.dataset_path), plan["indices"], tokenizer,
                       Agent._to_internal, **limits)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(suite, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "plan_sha256": sha256_file(args.plan), "suite_sha256": json_hash(suite),
                "cases": len(suite), "decisions": sum(len(p["questions"]) for p in suite),
                "source_indices": plan["indices"], "original_input_limits": limits,
                "generation": plan["generation"]}
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
