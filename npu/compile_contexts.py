#!/usr/bin/env python3
"""Precompile strict QNN contexts without loading Laya weights or embeddings.

Run on the target device with the same environment as the service::

    source npu/env.sh
    .venv/bin/python npu/compile_contexts.py --manifest npu/fidelity/manifest.json --verify-reload

The checkpoint and ONNX files are streamed through SHA256 verification; no
checkpoint tensor, tokenizer, upstream agent, or token embedding is loaded.
LAYA_NPU_CACHE_DIR selects the same persistent cache used by laya_npu.py.
"""

import argparse
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from laya_npu import QNNEncoder
from npu.fidelity_runtime import artifact_metadata, load_bucket_manifest


def _emit_json(record):
    print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)


def _config_only_encoder(model_dir):
    config_path = Path(model_dir) / "encoder/config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for name in ("hidden_size", "local_attention", "pad_token_id"):
        if not isinstance(config.get(name), int) or isinstance(config[name], bool):
            raise ValueError(f"{config_path}: {name} must be an integer")
    if config["hidden_size"] < 1 or config["local_attention"] < 0 or config["pad_token_id"] < 0:
        raise ValueError(f"{config_path}: invalid hidden_size, local_attention, or pad_token_id")
    # Session preparation never calls forward. Identity carries no parameters
    # and avoids allocating the checkpoint's large token embedding table.
    return SimpleNamespace(config=SimpleNamespace(**config), get_input_embeddings=torch.nn.Identity)


def _prepare_phase(adapter, buckets, phase, emit):
    timings = []
    for bucket in buckets:
        emit({"event": "bucket_start", "phase": phase, "bucket": bucket})
        start = time.perf_counter()
        previous_hits = adapter.stats["context_cache_hits"]
        # Use the same lock, strict provider options, cache key, and signature
        # validation as inference, without running the placeholder embedding.
        with adapter._lock:
            adapter._get_session(bucket)
        elapsed = time.perf_counter() - start
        cache_hit = adapter.stats["context_cache_hits"] > previous_hits
        if phase == "reload" and not cache_hit:
            raise RuntimeError(f"Bucket {bucket} did not reload its saved QNN context")
        record = {
            "event": "bucket_ready", "phase": phase, "bucket": bucket,
            "seconds": elapsed, "cache_hit": cache_hit,
            "cache_path": adapter.stats["context_cache_paths"][str(bucket)],
        }
        timings.append(record)
        emit(record)
    return {"buckets": timings, "stats": copy.deepcopy(adapter.stats)}


def compile_contexts(manifest_path, model_dir, verify_reload=False, emit=_emit_json):
    """Validate artifacts, prepare each bucket, and optionally reload every cache.

    Returns a JSON-serializable timing and provenance report. All HTP session
    references are released before the optional reload phase and before return.
    """
    started = time.perf_counter()
    manifest_path = Path(manifest_path).resolve()
    model_dir = Path(model_dir).resolve()
    bucket_paths, manifest = load_bucket_manifest(manifest_path)
    encoder = _config_only_encoder(model_dir)
    emit({"event": "artifact_verification_start", "manifest": str(manifest_path)})
    hash_start = time.perf_counter()
    metadata = artifact_metadata(
        manifest_path, manifest, bucket_paths, model_dir / "model.safetensors"
    )
    verification_seconds = time.perf_counter() - hash_start
    hashes = {int(bucket): item["sha256"] for bucket, item in metadata["models"].items()}
    emit({"event": "artifacts_verified", "seconds": verification_seconds,
          "buckets": list(bucket_paths), "checkpoint_sha256": metadata["checkpoint_sha256"]})

    def new_adapter():
        adapter = QNNEncoder(
            encoder, bucket_paths, manifest.get("mask_penalty", -100.0), model_hashes=hashes,
            zero_pad_embeddings=manifest.get("zero_pad_embeddings", False),
            correction_metadata=manifest.get("htp_conv_offset_correction"),
        )
        if not adapter.cache_enabled:
            raise ValueError("Context precompilation requires LAYA_NPU_CONTEXT_CACHE=1")
        return adapter

    adapter = new_adapter()
    try:
        preparation = _prepare_phase(adapter, bucket_paths, "prepare", emit)
    finally:
        del adapter
        gc.collect()

    reload_result = None
    if verify_reload:
        # Cache generation completed in an earlier adapter whose session has
        # now been released; this verifies deserialization and input signatures.
        for item in preparation["buckets"]:
            if not Path(item["cache_path"]).is_file():
                raise FileNotFoundError(f"Generated QNN context is missing: {item['cache_path']}")
        adapter = new_adapter()
        try:
            reload_result = _prepare_phase(adapter, bucket_paths, "reload", emit)
        finally:
            del adapter
            gc.collect()

    report = {
        "event": "complete", "manifest": str(manifest_path), "model_dir": str(model_dir),
        "manifest_sha256": metadata["manifest_sha256"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "graph_sha256": {str(bucket): digest for bucket, digest in hashes.items()},
        "artifact_verification_seconds": verification_seconds,
        "prepare": preparation, "reload": reload_result,
        "total_seconds": time.perf_counter() - started,
    }
    emit(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Precompile all manifest buckets without loading checkpoint tensors.",
        epilog="On the target, first run: source npu/env.sh",
    )
    parser.add_argument("--manifest", type=Path, default=Path(os.environ.get(
        "LAYA_NPU_MANIFEST", ROOT / "npu/fidelity/manifest.json"
    )))
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/multilingual")
    parser.add_argument("--verify-reload", action="store_true",
                        help="Release compilation sessions and verify every saved context loads again.")
    args = parser.parse_args(argv)
    compile_contexts(args.manifest, args.model_dir, args.verify_reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
