"""Pure input/mask helpers shared by fidelity exports and the HTP runtime."""

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np


def select_bucket(buckets, sequence_length):
    """Select capacity without changing or truncating the original sequence."""
    if sequence_length < 1:
        raise ValueError("Encoder sequences must contain at least one token")
    for bucket in sorted(buckets):
        if bucket >= sequence_length:
            return bucket
    maximum = max(buckets, default=0)
    raise ValueError(
        f"Sequence contains {sequence_length} tokens, but the largest HTP bucket is {maximum}. "
        "Build a larger fidelity bucket; input truncation and CPU fallback are disabled."
    )


def attention_masks(bucket, sequence_length, local_radius=64, mask_penalty=-100.0):
    """Finite additive masks matching ModernBERT's global/local visibility.

    Local attention includes positions at exactly local_radius distance. Padding
    and local exclusion are combined as a union, never two additive penalties.
    """
    if not 1 <= sequence_length <= bucket:
        raise ValueError("sequence_length must be between 1 and bucket capacity")
    if local_radius < 0:
        raise ValueError("local_radius must not be negative")
    if not math.isfinite(mask_penalty) or mask_penalty >= 0:
        raise ValueError("mask_penalty must be finite and negative")
    positions = np.arange(bucket)
    valid_keys = positions < sequence_length
    global_mask = np.broadcast_to(valid_keys[None, :], (bucket, bucket))
    local_mask = global_mask & (np.abs(positions[:, None] - positions[None, :]) <= local_radius)
    return tuple(
        np.where(visible, 0.0, mask_penalty).astype(np.float32)[None, None, :, :]
        for visible in (global_mask, local_mask)
    )


def load_bucket_manifest(path):
    """Resolve static model paths relative to the build's manifest."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Fidelity model manifest is missing: {path}. "
            "Build the fidelity ONNX models or set LAYA_NPU_MANIFEST; "
            "the legacy 64-token clipped model is not a fidelity-compatible default."
        )
    metadata = json.loads(path.read_text(encoding="utf-8"))
    entries = metadata.get("buckets")
    if isinstance(entries, dict):
        pairs = entries.items()
    elif isinstance(entries, list):
        pairs = [(entry["sequence_length"], entry["path"]) for entry in entries]
    else:
        raise ValueError("Fidelity manifest must contain a nonempty 'buckets' mapping or list")
    paths = {}
    for length, filename in pairs:
        length = int(length)
        if length < 1 or length in paths:
            raise ValueError(f"Invalid or duplicate manifest bucket: {length}")
        model_path = Path(filename)
        if not model_path.is_absolute():
            model_path = path.parent / model_path
        model_path = model_path.resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Manifest bucket {length} model is missing: {model_path}")
        paths[length] = model_path
    if not paths:
        raise ValueError("Fidelity manifest must include at least one bucket")
    penalty = float(metadata.get("mask_penalty", -100.0))
    if not math.isfinite(penalty) or penalty >= 0:
        raise ValueError("Manifest mask_penalty must be finite and negative")
    return dict(sorted(paths.items())), metadata


def sha256_file(path):
    """Hash large artifacts with bounded memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_digest(expected, actual, label):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError(f"Manifest {label} must be a SHA256 digest of 64 hexadecimal characters")
    if expected.lower() != actual:
        raise ValueError(f"Manifest {label} mismatch: expected {expected.lower()}, actual {actual}")


def artifact_metadata(manifest_path, manifest, bucket_paths, checkpoint_path):
    """Fingerprint and validate artifacts once when an agent is constructed.

    Optional manifest hashes bind exported graphs to the checkpoint supplying the
    tokenizer, embeddings and heads. A hash mismatch fails before model loading.
    """
    manifest_path = Path(manifest_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"The HTP runtime requires a local checkpoint file: {checkpoint_path}. "
            "Download the checkpoint used to export these graphs first."
        )
    checkpoint_hash = sha256_file(checkpoint_path)
    expected_checkpoint = manifest.get("checkpoint_sha256")
    if expected_checkpoint is not None:
        _check_digest(expected_checkpoint, checkpoint_hash, "checkpoint_sha256")

    expected_models = manifest.get("model_sha256", {})
    if isinstance(expected_models, str) and len(bucket_paths) == 1:
        expected_models = {str(next(iter(bucket_paths))): expected_models}
    if not isinstance(expected_models, dict):
        raise ValueError("Manifest model_sha256 must map bucket lengths to SHA256 digests")
    expected_models = {str(key): value for key, value in expected_models.items()}
    unknown = set(expected_models) - {str(length) for length in bucket_paths}
    if unknown:
        raise ValueError(f"Manifest model_sha256 declares unknown buckets: {sorted(unknown)}")

    unique_hashes = {}
    models = {}
    for length, path in sorted(bucket_paths.items()):
        path = Path(path).resolve()
        if path not in unique_hashes:
            unique_hashes[path] = sha256_file(path)
        digest = unique_hashes[path]
        expected = expected_models.get(str(length))
        if expected is not None:
            _check_digest(expected, digest, f"model_sha256[{length}]")
        models[str(length)] = {
            "path": str(path), "sha256": digest, "manifest_verified": expected is not None,
        }

    canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    provenance = {
        "manifest_path": str(manifest_path),
        "manifest": json.loads(canonical),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_canonical_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "supported_buckets": sorted(bucket_paths),
        "models": models,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_manifest_verified": expected_checkpoint is not None,
    }
    for field in ("precision", "activation_bits", "weight_bits", "source_checkpoint"):
        if field in manifest:
            provenance[field] = manifest[field]
    return provenance
