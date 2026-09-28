"""Pure input/mask helpers shared by fidelity exports and the HTP runtime."""

import hashlib
import json
import math
import os
import platform
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
    correction_runtime_requirements(metadata, paths)
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


def correction_runtime_requirements(manifest, bucket_paths):
    """Validate backend-specific correction provenance without loading QNN.

    Older corrections record only the backend hash. Additional stub/skeleton
    hashes are enforced when present, never inferred from a later installation.
    """
    if "htp_conv_offset_correction" not in manifest:
        return {}
    entries = manifest["htp_conv_offset_correction"]
    if not isinstance(entries, dict) or not entries:
        raise ValueError("htp_conv_offset_correction must be a nonempty bucket mapping")
    known = {str(bucket) for bucket in bucket_paths}
    requirements = {}
    for bucket, entry in entries.items():
        if str(bucket) not in known or str(int(bucket)) != str(bucket):
            raise ValueError(f"htp_conv_offset_correction declares unknown bucket {bucket}")
        if not isinstance(entry, dict) or not isinstance(entry.get("runtime"), dict):
            raise ValueError(f"Correction bucket {bucket} requires a runtime fingerprint")
        runtime = entry["runtime"]
        for field in ("onnxruntime", "onnxruntime_qnn"):
            if not isinstance(runtime.get(field), str) or not runtime[field].strip():
                raise ValueError(f"Correction bucket {bucket} requires runtime {field}")
        if type(runtime.get("htp_arch")) is not int or runtime["htp_arch"] < 1:
            raise ValueError(f"Correction bucket {bucket} requires an integer runtime htp_arch")
        if runtime.get("cpu_ep_fallback") is not False:
            raise ValueError(f"Correction bucket {bucket} must record cpu_ep_fallback=false")
        for field in ("backend_sha256", "stub_sha256", "skel_sha256"):
            if field == "backend_sha256" or field in runtime:
                expected = runtime.get(field)
                if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
                    raise ValueError(f"Correction bucket {bucket} requires a valid runtime {field}")
        output_hash = entry.get("output_sha256")
        if not isinstance(output_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", output_hash):
            raise ValueError(f"Correction bucket {bucket} requires output_sha256")
        requirements[int(bucket)] = dict(runtime)
    return requirements


def qnn_backend_fingerprint(onnxruntime_version, onnxruntime_qnn_version, backend_path,
                            htp_arch=68, *, require_auxiliary=False):
    """Hash the configured HTP binaries without constructing an NPU session.

    Use npu/env.sh for an unambiguous loader setup. Stub lookup follows the
    configured host search path, and skeleton lookup follows ADSP_LIBRARY_PATH.
    An absent explicit search path does not prove which system-default binary
    the loader would choose; no package-local fallback hash is invented. Paths
    are provenance, while validation and cache identity compare binary contents.
    """
    backend = Path(backend_path).resolve(strict=True)
    arch = int(htp_arch)
    result = {
        "onnxruntime": str(onnxruntime_version),
        "onnxruntime_qnn": str(onnxruntime_qnn_version),
        "backend_path": str(backend), "backend_sha256": sha256_file(backend),
        "platform": platform.platform(), "htp_arch": arch, "cpu_ep_fallback": False,
    }
    for label, filename, variable, separator in (
        ("stub", f"libQnnHtpV{arch}Stub.so", "LD_LIBRARY_PATH", os.pathsep),
        ("skel", f"libQnnHtpV{arch}Skel.so", "ADSP_LIBRARY_PATH", ";"),
    ):
        search_path = os.environ.get(variable, "")
        directories = [Path(item) if item else Path.cwd() for item in search_path.split(separator)] if search_path else []
        candidate = next((directory / filename for directory in directories
                          if (directory / filename).is_file()), None)
        if candidate is None:
            if require_auxiliary:
                raise FileNotFoundError(f"Cannot fingerprint {filename}; source npu/env.sh before calibration")
            continue
        candidate = candidate.resolve(strict=True)
        result[f"{label}_path"] = str(candidate)
        result[f"{label}_sha256"] = sha256_file(candidate)
    return result


def qnn_backend_identity(fingerprint):
    """Relocation-independent identity for quantization corrections and caches."""
    return {field: fingerprint[field] for field in (
        "onnxruntime", "onnxruntime_qnn", "htp_arch", "backend_sha256",
        "stub_sha256", "skel_sha256",
    ) if field in fingerprint}


def validate_correction_backend(expected, actual, bucket):
    """Fail before loading source/cached graphs when correction assumptions differ."""
    mismatches = []
    for field, value in qnn_backend_identity(expected).items():
        actual_value = actual.get(field)
        if field.endswith("_sha256"):
            value = value.lower()
            actual_value = actual_value.lower() if isinstance(actual_value, str) else actual_value
        if value != actual_value:
            mismatches.append(f"{field}: expected {value}, actual {actual_value}")
    if mismatches:
        raise RuntimeError(
            f"HTP Conv correction backend mismatch for bucket {bucket}: {'; '.join(mismatches)}. "
            "Restore the calibrated ORT/QNN binaries and source npu/env.sh, or recalibrate the "
            "uncorrected graph on this backend. Disabling context caching cannot bypass this check."
        )


def artifact_metadata(manifest_path, manifest, bucket_paths, checkpoint_path):
    """Fingerprint and validate artifacts once when an agent is constructed.

    Optional manifest hashes bind exported graphs to the checkpoint supplying the
    tokenizer, embeddings and heads. A hash mismatch fails before model loading.
    """
    correction_runtime_requirements(manifest, bucket_paths)
    corrections = manifest.get("htp_conv_offset_correction", {})
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
        if str(length) in corrections:
            _check_digest(corrections[str(length)]["output_sha256"], digest,
                          f"htp_conv_offset_correction[{length}].output_sha256")
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
