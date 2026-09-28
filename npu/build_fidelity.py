"""Export the original ModernBERT function and calibrate a QNN 16a8w graph.

Run on x86/WSL; the resulting ONNX graph is portable to the ARM QNN runtime.
Calibration indices are explicit and must be excluded from held-out evaluation.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from contextlib import nullcontext

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USE_TF", "0")
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config
import laya
from laya.common import build_sequence

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_input_hashes(model_dir, parquet):
    """Bind export reuse to calibration data, configuration and tokenizer bytes."""
    model_dir = Path(model_dir)
    required = ["rl_agent_config.json", "encoder/config.json",
                "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]
    for name in required:
        if not (model_dir / name).is_file():
            raise FileNotFoundError(f"Export input is missing: {model_dir / name}")
    paths = {model_dir / name for name in required}
    paths.update(path for path in (model_dir / "tokenizer").rglob("*") if path.is_file())
    return {
        "calibration_dataset_sha256": sha256_file(parquet),
        "model_files_sha256": {path.relative_to(model_dir).as_posix(): sha256_file(path)
                               for path in sorted(paths)},
    }


def preflight_build(args, fp32, qdq, export_config):
    """Reject conflicting policies and existing outputs before any build writes.

    Reuse consumes a previously verified FP32 graph but produces a new QDQ file.
    A metadata-only file left by --export-only may be finalized during reuse.
    Frozen or partially written QDQ graphs always require a new output directory.
    """
    manifest_path = args.output_dir / "manifest.json"
    policy = {
        "checkpoint_sha256": export_config["checkpoint_sha256"],
        "mask_penalty": args.mask_penalty, "precision": f"a{args.activation_bits}w8",
        "zero_pad_embeddings": args.zero_pad_embeddings,
        "matmul_rhs_refinement": args.refine_matmul_rhs,
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        defaults = {"zero_pad_embeddings": False, "matmul_rhs_refinement": "none"}
        for field, expected in policy.items():
            if manifest.get(field, defaults.get(field)) != expected:
                raise ValueError(f"Output manifest has a different {field} policy; choose a new output directory")
        if "htp_conv_offset_correction" in manifest:
            raise ValueError("Do not build into a corrected model manifest; choose a new output directory")
        if not isinstance(manifest.get("buckets"), dict) or not isinstance(manifest.get("model_sha256"), dict):
            raise ValueError("Build output manifest requires buckets and model_sha256 mappings")
        if str(args.length) in manifest["buckets"]:
            raise FileExistsError(f"Output manifest already contains bucket {args.length}; choose a new output directory")
    else:
        manifest = {**policy, "buckets": {}, "model_sha256": {}}
    if qdq.exists():
        raise FileExistsError(f"Quantized output already exists: {qdq}; choose a new output directory")
    sidecar = fp32.with_suffix(".json")
    if args.reuse_export:
        if not fp32.is_file() or not sidecar.is_file():
            raise FileNotFoundError("--reuse-export requires both the FP32 graph and its export sidecar")
        saved = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(saved.get("config"), dict) or "input_sha256" not in saved["config"]:
            raise ValueError("Legacy export sidecar lacks input hashes; make a fresh export in a new output directory")
        if saved["config"] != export_config or saved.get("fp32_sha256") != sha256_file(fp32):
            raise ValueError("Existing export does not match the requested configuration/checkpoint/input hashes")
    else:
        for path in (fp32, sidecar, qdq.with_suffix(".json")):
            if path.exists():
                raise FileExistsError(
                    f"Build output already exists: {path}; use --reuse-export for a verified FP32 export "
                    "or choose a new output directory"
                )
    return manifest


def masks(length, valid, radius, penalty):
    positions = np.arange(length)
    global_mask = np.broadcast_to(positions[None, :] >= valid, (length, length))
    local_mask = global_mask | (abs(positions[:, None] - positions[None, :]) > radius)
    return [(x.astype(np.float32) * penalty)[None, None] for x in (global_mask, local_mask)]


class Backbone(nn.Module):
    def __init__(self, encoder, length, split_cls=False):
        super().__init__()
        self.encoder = encoder
        self.split_cls = split_cls
        # Eager attention emits ordinary ONNX MatMul, Softmax and native Gelu.
        self.encoder.config._attn_implementation = "eager"
        self.encoder.config.reference_compile = False
        for kind in set(encoder.config.layer_types):
            cos, sin = encoder.rotary_emb(torch.zeros(1, length, encoder.config.hidden_size), torch.arange(length)[None], kind)
            self.register_buffer("cos_" + kind, cos)
            self.register_buffer("sin_" + kind, sin)

    def forward(self, inputs_embeds, attn_mask, sliding_mask):
        h = self.encoder.embeddings.norm(inputs_embeds)
        if self.split_cls:
            # The CLS token is a learned activation sink. Keep its residual
            # quantization range separate from the remaining tokens. Attention
            # still sees every token; concatenate only normalized representations.
            cls, rest = h[:, :1], h[:, 1:]
            for layer in self.encoder.layers:
                normalized = torch.cat((layer.attn_norm(cls), layer.attn_norm(rest)), dim=1)
                attention = layer.attn(normalized, attention_mask=attn_mask, sliding_window_mask=sliding_mask, position_ids=None,
                                       position_embeddings=(getattr(self, "cos_" + layer.attention_type), getattr(self, "sin_" + layer.attention_type)))[0]
                cls, rest = cls + attention[:, :1], rest + attention[:, 1:]
                cls = cls + layer.mlp(layer.mlp_norm(cls))
                rest = rest + layer.mlp(layer.mlp_norm(rest))
            return torch.cat((self.encoder.final_norm(cls), self.encoder.final_norm(rest)), dim=1)
        for layer in self.encoder.layers:
            h = layer(h, attention_mask=attn_mask, sliding_window_mask=sliding_mask,
                      position_embeddings=(getattr(self, "cos_" + layer.attention_type), getattr(self, "sin_" + layer.attention_type)))[0]
        return self.encoder.final_norm(h)


def sequences(agent, parquet, indices):
    import pyarrow.parquet as pq
    rows = pq.read_table(parquet).to_pylist()
    if (not indices or len(set(indices)) != len(indices)
            or any(type(index) is not int or not 0 <= index < len(rows) for index in indices)):
        raise ValueError("Calibration indices must be distinct valid nonnegative row indices")
    result = []
    for index in indices:
        row = rows[index]
        state, questions = json.loads(row["state"]), json.loads(row["questions"])
        for qid, q in questions.items():
            ids, _ = build_sequence(agent.tok, state, agent._to_internal(q), agent.cfg["max_len"], agent.cfg["head_max_len"])
            result.append((index, qid, ids))
    return result


def long_calibration_sequences(agent, parquet, selected, target_length):
    """Repeat only reserved calibration states to exercise full valid length.

    Questions, criteria, tokenizer and upstream sequence/head budgets are kept.
    These are additional calibration inputs, never held-out evaluation examples.
    """
    import pyarrow.parquet as pq
    if target_length != agent.cfg["max_len"]:
        raise ValueError("Long calibration requires the original max_len bucket")
    rows = pq.read_table(parquet).to_pylist()
    result = []
    for index, qid, _ in selected:
        if type(index) is not int or not 0 <= index < len(rows):
            raise ValueError("Selected calibration index must be a valid nonnegative row index")
        row = rows[index]
        state, questions = json.loads(row["state"]), json.loads(row["questions"])
        if not isinstance(state, (str, dict, list)) or not state:
            raise ValueError(f"Cannot extend empty or unsupported calibration state at row {index}")
        repeated = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        if not repeated.strip():
            raise ValueError(f"Cannot extend empty calibration state at row {index}")
        for _ in range(12):
            ids, _ = build_sequence(agent.tok, repeated, agent._to_internal(questions[qid]),
                                    agent.cfg["max_len"], agent.cfg["head_max_len"])
            if len(ids) == target_length:
                result.append((index, qid + "/repeated-state", ids))
                break
            repeated = repeated + "\n\n" + repeated
        else:
            raise ValueError(f"Calibration state did not fill {target_length} tokens at row {index}")
    return result


@torch.no_grad()
def channel_ranges(encoder, seqs):
    maxima = [torch.zeros(layer.mlp.Wo.in_features) for layer in encoder.layers]
    handles = []
    for i, layer in enumerate(encoder.layers):
        def capture(module, args, index=i):
            maxima[index] = torch.maximum(maxima[index], args[0].abs().amax(dim=(0, 1)).cpu())
        handles.append(layer.mlp.Wo.register_forward_pre_hook(capture))
    for _, _, ids in seqs:
        tensor = torch.tensor([ids])
        encoder(input_ids=tensor, attention_mask=torch.ones_like(tensor))
    for handle in handles:
        handle.remove()
    return maxima


@torch.no_grad()
def balance_mlp(encoder, maxima):
    """Exact diagonal reparameterization; no clipping or retraining."""
    stats = []
    for layer, a in zip(encoder.layers, maxima):
        mlp = layer.mlp
        w = mlp.Wo.weight.abs().amax(dim=0).clamp_min(1e-5)
        scale = (a.clamp_min(1e-5) / w).sqrt()
        scale = (scale / scale.median()).clamp(1 / 16, 256)
        size = len(scale)
        mlp.Wi.weight[size:].div_(scale[:, None])
        if mlp.Wi.bias is not None:
            mlp.Wi.bias[size:].div_(scale)
        mlp.Wo.weight.mul_(scale[None, :])
        stats.append({"activation_max_before": float(a.max()), "activation_max_after": float((a / scale).max()),
                      "scale_min": float(scale.min()), "scale_max": float(scale.max())})
    return stats


class Reader(CalibrationDataReader):
    def __init__(self, data):
        self.data = data
        self.rewind()
    def rewind(self):
        self.iterator = iter(self.data)
    def get_next(self):
        return next(self.iterator, None)
    def __len__(self):
        return len(self.data)
    def set_range(self, start_index, end_index):
        self.iterator = iter(self.data[start_index:end_index])


def fix_layernorm_bias(model):
    constants = {n.output[0]: n for n in model.graph.node if n.op_type == "Constant"}
    removed = set()
    for node in model.graph.node:
        if node.op_type == "LayerNormalization" and len(node.input) > 2 and node.input[2] in constants:
            const = constants[node.input[2]]
            for attr in const.attribute:
                if attr.name == "value":
                    value = onnx.TensorProto()
                    value.CopyFrom(attr.t)
                    value.name = node.input[2]
                    model.graph.initializer.append(value)
                    removed.add(const.name)
    keep = [n for n in model.graph.node if n.name not in removed]
    del model.graph.node[:]
    model.graph.node.extend(keep)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--length", type=int, default=768)
    p.add_argument("--model", type=Path, default=ROOT / "models/multilingual")
    p.add_argument("--parquet", type=Path, default=ROOT / ".work/typed-decisions.parquet")
    p.add_argument("--indices", default="0,25,50,75,100,125,150,175,200,225,250,275,300,325,350,375")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--append-long-calibration", action="store_true",
                   help="Append full-length repeated-state variants of the selected reserved calibration questions")
    p.add_argument("--output-dir", type=Path, default=ROOT / "npu/fidelity")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--mask-penalty", type=float, default=-100)
    p.add_argument("--balance", action="store_true")
    p.add_argument("--group-outliers", action="store_true")
    p.add_argument("--outlier-ratio", type=float, default=16)
    p.add_argument("--max-outliers", type=int, default=8)
    p.add_argument("--conv-linear", action="store_true", help="Use 1x1 convolution for per-channel weight quantization")
    p.add_argument("--split-cls", action="store_true", help="Keep CLS and other token residual quantization ranges separate")
    p.add_argument("--fold-norms", action="store_true", help="Exactly fold interior normalization affine weights into projections")
    p.add_argument("--zero-pad-embeddings", action="store_true", help="Zero masked padding embeddings so they do not dominate calibration ranges")
    p.add_argument("--refine-matmul-rhs", choices=["none", "pv", "all"], default="none",
                   help="Approximate dynamic U16 RHS with two U8 MatMul terms (experimental HTP refinement)")
    p.add_argument("--export-only", action="store_true")
    p.add_argument("--reuse-export", action="store_true",
                   help="Reuse a verified FP32 export to create a new QDQ graph; existing QDQ outputs are rejected")
    p.add_argument("--low-memory-calibration", action="store_true",
                   help="Disable calibration memory arena and merge MinMax ranges after every sample")
    p.add_argument("--activation-bits", type=int, choices=[8, 16], default=16)
    args = p.parse_args()
    if args.length < 1 or args.samples < 1:
        p.error("length and samples must be positive")
    if args.refine_matmul_rhs != "none" and args.activation_bits != 16:
        p.error("MatMul RHS refinement requires --activation-bits 16")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    fp32 = args.output_dir / f"backbone_{args.length}_fp32.onnx"
    qdq = args.output_dir / f"backbone_{args.length}_a{args.activation_bits}w8.onnx"
    checkpoint_hash = sha256_file(args.model / "model.safetensors")
    export_config = {key: getattr(args, key) for key in ("length", "indices", "samples", "mask_penalty", "balance", "group_outliers", "outlier_ratio", "max_outliers", "conv_linear", "split_cls", "fold_norms", "zero_pad_embeddings")}
    export_config["checkpoint_sha256"] = checkpoint_hash
    export_config["input_sha256"] = export_input_hashes(args.model, args.parquet)
    if args.append_long_calibration:
        export_config["append_long_calibration"] = True
    export_sidecar = fp32.with_suffix(".json")
    manifest = preflight_build(args, fp32, qdq, export_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("Loading pristine LAYA", flush=True)
    agent = laya.load(str(args.model), device="cpu")
    enc = agent.model.encoder
    seqs_all = sequences(agent, args.parquet, [int(i) for i in args.indices.split(",")])
    # Bucket calibration only uses complete upstream sequences that fit it.
    seqs_all = [entry for entry in seqs_all if len(entry[2]) <= args.length]
    if not seqs_all:
        raise ValueError("No complete calibration sequences fit the requested bucket")
    # Spread limited calibration across every selected case/domain, not only the first workflow.
    positions = np.linspace(0, len(seqs_all) - 1, min(args.samples, len(seqs_all)), dtype=int)
    seqs = [seqs_all[i] for i in positions]
    base_count = len(seqs)
    if args.append_long_calibration:
        seqs += long_calibration_sequences(agent, args.parquet, seqs, args.length)
    metadata = {"sequence_length": args.length, "mask_penalty": args.mask_penalty, "calibration_cases": sorted(set(i for i, _, _ in seqs)),
                "calibration_questions": len(seqs), "activation_bits": args.activation_bits, "weight_bits": 8, "balanced": args.balance,
                "zero_pad_embeddings": args.zero_pad_embeddings}
    metadata["base_calibration_questions"] = base_count
    metadata["calibration_dataset_sha256"] = export_config["input_sha256"]["calibration_dataset_sha256"]
    metadata["original_input_limits"] = {key: agent.cfg[key] for key in ("max_len", "head_max_len")}
    metadata["long_calibration_questions"] = len(seqs) - base_count
    metadata["calibration_valid_lengths"] = [len(ids) for _, _, ids in seqs]
    metadata["calibration_question_identities"] = [[index, qid] for index, qid, _ in seqs]
    # Save untouched reference outputs before any exact reparameterization.
    validation = []
    validation_indices = list(range(min(2, base_count)))
    if args.append_long_calibration:
        validation_indices += list(range(base_count, base_count + min(2, base_count)))
    for index in validation_indices:
        _, _, ids = seqs[index]
        tensor = torch.tensor([ids])
        validation.append(enc(input_ids=tensor, attention_mask=torch.ones_like(tensor)).last_hidden_state.clone())
    if args.fold_norms and not args.reuse_export:
        sys.path.insert(0, str(ROOT))
        from npu.norm_folding import fold_interior_norms
        metadata["norm_folding"] = fold_interior_norms(enc)
    if args.balance and not args.reuse_export:
        print("Collecting per-channel GeGLU ranges", flush=True)
        metadata["balance"] = balance_mlp(enc, channel_ranges(enc, seqs))
    if args.group_outliers and not args.reuse_export:
        sys.path.insert(0, str(ROOT))
        from npu.grouped_geglu import GroupedGeGLU
        print("Separating GeGLU outlier channels", flush=True)
        ranges = channel_ranges(enc, seqs)
        groups = []
        for layer, maxima in zip(enc.layers, ranges):
            layer.mlp = GroupedGeGLU.from_mlp(layer.mlp, maxima, outlier_ratio=args.outlier_ratio, max_outliers=args.max_outliers)
            groups.append({**layer.mlp.summary(), "activation_max": float(maxima.max())})
        metadata["groups"] = groups
    if args.conv_linear and not args.reuse_export:
        sys.path.insert(0, str(ROOT))
        from npu.conv_linear import replace_linears_with_conv
        metadata["conv_linear_replacements"] = replace_linears_with_conv(enc)
    wrapper = Backbone(enc, args.length, args.split_cls).eval()
    data = []
    for _, _, ids in seqs:
        padded = ids + [agent.tok.pad_token_id] * (args.length - len(ids))
        attn, sliding = masks(args.length, len(ids), enc.config.local_attention // 2, args.mask_penalty)
        embeds = enc.embeddings.tok_embeddings(torch.tensor([padded])).numpy()
        if args.zero_pad_embeddings:
            embeds[:, len(ids):] = 0
        data.append({"inputs_embeds": embeds, "attn_mask": attn, "sliding_mask": sliding})
    metadata["torch_export_error"] = []
    validation_data = [data[index] for index in validation_indices]
    for d, ref in zip(validation_data, validation):
        actual = wrapper(*(torch.from_numpy(d[k]) for k in ("inputs_embeds", "attn_mask", "sliding_mask")))[:, :ref.shape[1]]
        error = (actual - ref).abs()
        metadata["torch_export_error"].append({"mean_abs": float(error.mean()), "max_abs": float(error.max())})
    print("FP32 wrapper checks", metadata["torch_export_error"], flush=True)
    if not args.reuse_export:
        print("Exporting", fp32, flush=True)
        torch.onnx.export(wrapper, tuple(torch.from_numpy(data[0][k]) for k in ("inputs_embeds", "attn_mask", "sliding_mask")), str(fp32),
                          input_names=["inputs_embeds", "attn_mask", "sliding_mask"], output_names=["last_hidden_state"], opset_version=20,
                          do_constant_folding=True, dynamo=False)
        model = onnx.load(fp32)
        fix_layernorm_bias(model)
        onnx.save(model, fp32)
        export_sidecar.write_text(json.dumps({"config": export_config, "fp32_sha256": sha256_file(fp32)}, indent=2) + "\n")
        del model
    del wrapper, enc, agent
    gc.collect()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(fp32), opts, providers=["CPUExecutionProvider"])
    metadata["onnx_export_error"] = []
    for d, ref in zip(validation_data, validation):
        actual = sess.run(None, d)[0][:, :ref.shape[1]]
        error = abs(actual - ref.numpy())
        metadata["onnx_export_error"].append({"mean_abs": float(error.mean()), "max_abs": float(error.max())})
    print("ONNX export checks", metadata["onnx_export_error"], flush=True)
    if any(not np.isfinite(x["max_abs"]) or x["max_abs"] > 0.01 for x in metadata["onnx_export_error"]):
        raise RuntimeError("FP32 export differs from original; refusing to quantize")
    del sess
    gc.collect()
    if not args.export_only:
        model = onnx.load(fp32)
        initializers = {x.name for x in model.graph.initializer}
        overrides = {}
        if args.activation_bits == 16:
            for node in model.graph.node:
                if node.op_type == "MatMul" and node.input[1] not in initializers:
                    overrides[node.input[1]] = [{"quant_type": QuantType.QUInt16, "convert": {"quant_type": QuantType.QUInt8, "recv_nodes": {node.name}}}]
        if args.conv_linear:
            for node in model.graph.node:
                if node.op_type == "Conv" and node.input[1] in initializers:
                    overrides[node.input[1]] = [{"quant_type": QuantType.QInt8, "axis": 0, "symmetric": True}]
        config = get_qnn_qdq_config(model, Reader(data), activation_type=QuantType.QUInt16 if args.activation_bits == 16 else QuantType.QUInt8,
                                    weight_type=QuantType.QUInt8, init_overrides=overrides, add_qtype_converts=False, per_channel=args.conv_linear)
        del model
        gc.collect()
        print("Calibrating and quantizing", qdq, flush=True)
        memory_context = nullcontext()
        if args.low_memory_calibration:
            sys.path.insert(0, str(ROOT))
            from npu.calibration_memory import low_memory_calibration
            config.extra_options["CalibStridedMinMax"] = 1
            memory_context = low_memory_calibration(args.threads)
            metadata["calibration_memory"] = "no CPU arena/pattern; exact MinMax merge every sample"
        with memory_context:
            quantize(str(fp32), str(qdq), config)
        if args.refine_matmul_rhs != "none":
            sys.path.insert(0, str(ROOT))
            from npu.matmul_refinement import attention_matmul_refinement
            model = onnx.load(qdq)
            metadata["matmul_rhs_refinement"] = attention_matmul_refinement(model, scope=args.refine_matmul_rhs)
            if not metadata["matmul_rhs_refinement"]["refined_matmuls"]:
                raise RuntimeError("The requested MatMul refinement did not match any dynamic U16-to-U8 RHS")
            onnx.checker.check_model(model)
            onnx.save(model, qdq)
            del model
        if args.fold_norms and args.activation_bits == 16:
            sys.path.insert(0, str(ROOT))
            from npu.unit_layernorm import repair_unit_layernorm_gamma
            model = onnx.load(qdq)
            metadata["static_unit_layernorm"] = repair_unit_layernorm_gamma(model)
            onnx.checker.check_model(model)
            onnx.save(model, qdq)
            del model
        metadata["model_sha256"] = sha256_file(qdq)
        manifest_path = args.output_dir / "manifest.json"
        manifest["buckets"][str(args.length)] = qdq.name
        manifest["model_sha256"][str(args.length)] = metadata["model_sha256"]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    metadata["build_seconds"] = time.time() - started
    qdq.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
