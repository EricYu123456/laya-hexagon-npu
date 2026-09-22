#!/usr/bin/env python3
"""Comprehensive benchmark of Laya on LocalLLaMA/typed-decisions (400 cases / 2,000 decisions).
Compares Qualcomm Hexagon NPU (HTP V68 QDQ UINT8) against CPU PyTorch (FP32 baseline).
"""

import json
import time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tqdm import tqdm

import laya
import laya_npu

ROOT = Path(__file__).resolve().parent

def main():
    print("=" * 70)
    print("Laya Benchmark: Qualcomm Hexagon HTP vs CPU PyTorch")
    print("Dataset: LocalLLaMA/typed-decisions (400 cases, 2,000 decisions)")
    print("=" * 70)

    # 1. Download Dataset
    print("\n[1/4] Loading dataset from Hugging Face...")
    parquet_path = hf_hub_download(
        "LocalLLaMA/typed-decisions",
        filename="all/test-00000-of-00001.parquet",
        repo_type="dataset"
    )
    table = pq.read_table(parquet_path)
    total_cases = len(table)
    print(f"Loaded {total_cases} test cases from {parquet_path}")

    # 2. Load Models
    print("\n[2/4] Loading models...")
    print("  • Loading PyTorch CPU Agent (multilingual)...")
    cpu_agent = laya.load(str(ROOT / "models/multilingual"), device="cpu")

    print("  • Loading Qualcomm Hexagon NPU Agent (modernbert_clamped_qdq.onnx)...")
    npu_agent = laya_npu.load(str(ROOT / "models/multilingual"), device="npu")

    # Warmup NPU
    print("  • Warming up Hexagon HTP...")
    sample_state = json.loads(table["state"][0].as_py())
    sample_q = json.loads(table["questions"][0].as_py())
    for _ in range(3):
        npu_agent.predict(sample_state, sample_q)

    # 3. Run Benchmark
    print(f"\n[3/4] Evaluating {total_cases} test cases (2,000 decisions)...")

    results = {
        "dataset": "LocalLLaMA/typed-decisions",
        "total_cases": total_cases,
        "total_decisions": 0,
        "cpu": {"correct": 0, "latencies_ms": []},
        "npu": {"correct": 0, "latencies_ms": []},
        "agreement": 0,
        "workflows": {},
        "qtypes": {"choice": {"total": 0, "agree": 0, "cpu_corr": 0, "npu_corr": 0},
                   "score":  {"total": 0, "agree": 0, "cpu_corr": 0, "npu_corr": 0},
                   "noul":   {"total": 0, "agree": 0, "cpu_corr": 0, "npu_corr": 0}},
    }

    start_all = time.perf_counter()

    for i in tqdm(range(total_cases), desc="Evaluating"):
        row = {k: table[k][i].as_py() for k in table.column_names}
        wf = row.get("workflow", "unknown")
        if wf not in results["workflows"]:
            results["workflows"][wf] = {"cases": 0, "decisions": 0, "agree": 0, "cpu_corr": 0, "npu_corr": 0}
        results["workflows"][wf]["cases"] += 1

        state = json.loads(row["state"]) if isinstance(row["state"], str) else row["state"]
        questions = json.loads(row["questions"]) if isinstance(row["questions"], str) else row["questions"]
        gold = json.loads(row["gold"]) if isinstance(row["gold"], str) else row["gold"]

        # Run NPU
        t0 = time.perf_counter()
        npu_res = npu_agent.predict(state, questions)
        npu_ms = (time.perf_counter() - t0) * 1000
        results["npu"]["latencies_ms"].append(npu_ms)

        # Run CPU
        t0 = time.perf_counter()
        cpu_res = cpu_agent.predict(state, questions)
        cpu_ms = (time.perf_counter() - t0) * 1000
        results["cpu"]["latencies_ms"].append(cpu_ms)

        # Score decisions
        for qid, qdef in questions.items():
            results["total_decisions"] += 1
            results["workflows"][wf]["decisions"] += 1
            qt = qdef["type"]
            results["qtypes"][qt]["total"] += 1

            g = gold[qid]
            c_ans = cpu_res["answers"][qid]
            n_ans = npu_res["answers"][qid]

            if qt == "choice":
                c_lbl = c_ans["choice"]
                n_lbl = n_ans["choice"]
                g_lbl = g["label"]
            elif qt == "noul":
                c_lbl = "true" if c_ans["noul"] > 0.5 else "false"
                n_lbl = "true" if n_ans["noul"] > 0.5 else "false"
                g_lbl = g["label"]
            elif qt == "score":
                c_lbl = str(round(c_ans["score"]))
                n_lbl = str(round(n_ans["score"]))
                g_lbl = str(int(round(float(g["label"]))))

            # Compare against gold
            cpu_hit = (c_lbl == g_lbl)
            npu_hit = (n_lbl == g_lbl)
            agree_hit = (c_lbl == n_lbl)

            if cpu_hit:
                results["cpu"]["correct"] += 1
                results["workflows"][wf]["cpu_corr"] += 1
                results["qtypes"][qt]["cpu_corr"] += 1

            if npu_hit:
                results["npu"]["correct"] += 1
                results["workflows"][wf]["npu_corr"] += 1
                results["qtypes"][qt]["npu_corr"] += 1

            if agree_hit:
                results["agreement"] += 1
                results["workflows"][wf]["agree"] += 1
                results["qtypes"][qt]["agree"] += 1

    total_time = time.perf_counter() - start_all

    # 4. Compute Metrics
    n_dec = results["total_decisions"]
    cpu_acc = (results["cpu"]["correct"] / n_dec) * 100
    npu_acc = (results["npu"]["correct"] / n_dec) * 100
    agree_pct = (results["agreement"] / n_dec) * 100

    cpu_p50 = np.median(results["cpu"]["latencies_ms"])
    npu_p50 = np.median(results["npu"]["latencies_ms"])
    cpu_avg = np.mean(results["cpu"]["latencies_ms"])
    npu_avg = np.mean(results["npu"]["latencies_ms"])
    speedup = cpu_p50 / npu_p50

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY (400 Cases / 2,000 Decisions)")
    print("=" * 70)
    print(f"Total Evaluation Time: {total_time:.1f} seconds")
    print(f"\n[Accuracy & Agreement]")
    print(f"  • CPU PyTorch Accuracy vs Gold : {results['cpu']['correct']}/{n_dec} ({cpu_acc:.2f}%)")
    print(f"  • Hexagon NPU Accuracy vs Gold : {results['npu']['correct']}/{n_dec} ({npu_acc:.2f}%)")
    print(f"  • NPU vs CPU Decision Agreement: {results['agreement']}/{n_dec} ({agree_pct:.2f}%)")

    print(f"\n[Latency per 5-Question Case]")
    print(f"  • CPU PyTorch Latency : p50 = {cpu_p50:.1f} ms | avg = {cpu_avg:.1f} ms")
    print(f"  • Hexagon HTP Latency : p50 = {npu_p50:.1f} ms | avg = {npu_avg:.1f} ms")
    print(f"  • Hardware Speedup    : {speedup:.2f}x faster on Hexagon NPU!")

    print(f"\n[Breakdown by Question Type]")
    for qt, d in results["qtypes"].items():
        pct_agree = (d["agree"] / d["total"]) * 100 if d["total"] else 0
        pct_cpu = (d["cpu_corr"] / d["total"]) * 100 if d["total"] else 0
        pct_npu = (d["npu_corr"] / d["total"]) * 100 if d["total"] else 0
        print(f"  • {qt:<8}: Total={d['total']:<4} | CPU Acc={pct_cpu:5.1f}% | NPU Acc={pct_npu:5.1f}% | Agreement={pct_agree:5.1f}%")

    print(f"\n[Breakdown by Workflow]")
    for wf, d in results["workflows"].items():
        pct_agree = (d["agree"] / d["decisions"]) * 100 if d["decisions"] else 0
        pct_cpu = (d["cpu_corr"] / d["decisions"]) * 100 if d["decisions"] else 0
        pct_npu = (d["npu_corr"] / d["decisions"]) * 100 if d["decisions"] else 0
        print(f"  • {wf:<25}: Decisions={d['decisions']:<4} | CPU Acc={pct_cpu:5.1f}% | NPU Acc={pct_npu:5.1f}% | Agreement={pct_agree:5.1f}%")

    # Save to JSON
    summary_data = {
        "dataset": "LocalLLaMA/typed-decisions",
        "total_cases": total_cases,
        "total_decisions": n_dec,
        "accuracy": {
            "cpu_pytorch_percent": round(cpu_acc, 2),
            "npu_hexagon_percent": round(npu_acc, 2),
            "npu_cpu_agreement_percent": round(agree_pct, 2)
        },
        "latency_case_ms": {
            "cpu_p50": round(float(cpu_p50), 1),
            "cpu_avg": round(float(cpu_avg), 1),
            "npu_p50": round(float(npu_p50), 1),
            "npu_avg": round(float(npu_avg), 1),
            "speedup_x": round(float(speedup), 2)
        },
        "workflows": results["workflows"],
        "qtypes": results["qtypes"]
    }
    report_file = ROOT / "benchmark_typed_decisions.json"
    report_file.write_text(json.dumps(summary_data, indent=2) + "\n")
    print(f"\nReport written to: {report_file}")

if __name__ == "__main__":
    main()
