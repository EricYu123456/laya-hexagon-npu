"""Laya's original inference API with its encoder executed on Qualcomm HTP.

Tokenization, sequence budgets, decision heads, temperatures and output formatting
are owned by the installed upstream Laya Agent. Only the encoder is replaced.
"""

import copy
import gc
import hashlib
import importlib.metadata
import json
import os
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from npu.fidelity_runtime import artifact_metadata, attention_masks, load_bucket_manifest, select_bucket, sha256_file


_QNN_REGISTRATION_LOCK = threading.Lock()
_QNN_REGISTERED = False
_CONTEXT_CACHE_SCHEMA = 1


def _package_version(module, distribution):
    version = getattr(module, "__version__", None)
    if version:
        return str(version)
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"Cannot identify {distribution} version for the QNN context cache; "
            "install its package metadata or set LAYA_NPU_CONTEXT_CACHE=0"
        ) from exc


def _qnn_runtime():
    """Import the ARM-only provider only when an HTP session is first needed."""
    global _QNN_REGISTERED
    import onnxruntime as ort
    import onnxruntime_qnn as qnn

    with _QNN_REGISTRATION_LOCK:
        if not _QNN_REGISTERED:
            ort.register_execution_provider_library("QNNExecutionProvider", qnn.get_library_path())
            _QNN_REGISTERED = True
    return ort, qnn


class QNNEncoder(torch.nn.Module):
    """Bucketed, batch-one HTP backbone retaining the original token embeddings.

    Only one QNN session is retained, bounding HTP graph memory as input lengths
    change. Unsupported lengths and provider failures are errors, never CPU fallback.
    """

    def __init__(self, encoder, bucket_paths, mask_penalty=-100.0, *, model_hashes=None,
                 cache_dir=None, cache_enabled=None):
        super().__init__()
        self.config = encoder.config
        self.tok_embeddings = encoder.get_input_embeddings()
        self.bucket_paths = dict(sorted(bucket_paths.items()))
        self.mask_penalty = float(mask_penalty)
        self.local_radius = int(self.config.local_attention) // 2
        self.pad_token_id = int(self.config.pad_token_id)
        self.hidden_size = int(self.config.hidden_size)
        self._session = None
        self._session_bucket = None
        self._lock = threading.RLock()
        if cache_enabled is None:
            setting = os.environ.get("LAYA_NPU_CONTEXT_CACHE", "1").strip().lower()
            if setting not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
                raise ValueError("LAYA_NPU_CONTEXT_CACHE must be a boolean value, such as 0 or 1")
            cache_enabled = setting in {"1", "true", "yes", "on"}
        self.cache_enabled = bool(cache_enabled)
        self.cache_dir = Path(cache_dir or os.environ.get(
            "LAYA_NPU_CACHE_DIR", str(Path(__file__).resolve().parent / "npu/fidelity/context_cache")
        )).expanduser().resolve()
        # NPUAgent supplies hashes already verified by artifact_metadata. Direct
        # users are hashed lazily once per bucket, never on every inference.
        self._model_hashes = {int(bucket): digest for bucket, digest in (model_hashes or {}).items()}
        self.stats = {
            "npu_calls": 0,
            "cpu_fallbacks": 0,
            "session_creations": 0,
            "session_evictions": 0,
            "bucket_calls": {},
            "context_cache_enabled": self.cache_enabled,
            "context_cache_hits": 0,
            "context_cache_misses": 0,
            "context_cache_writes": 0,
            "context_cache_errors": 0,
            "context_cache_paths": {},
        }

    def get_input_embeddings(self):
        return self.tok_embeddings

    def _session_options(self, ort, devices, provider_options, *, generate_path=None, cached_path=None):
        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        if generate_path is not None:
            # Embedding the context makes publication an atomic single-file
            # rename, with no relative .bin paths to repair afterward.
            # https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html
            options.add_session_config_entry("ep.context_enable", "1")
            options.add_session_config_entry("ep.context_embed_mode", "1")
            options.add_session_config_entry("ep.context_file_path", str(generate_path))
        elif cached_path is not None:
            options.add_session_config_entry("ep.context_enable", "0")
            options.add_session_config_entry("ep.context_file_path", str(cached_path))
        options.add_provider_for_devices(devices, provider_options)
        return options

    def _context_cache_path(self, bucket, ort, qnn, provider_options):
        if bucket not in self._model_hashes:
            self._model_hashes[bucket] = sha256_file(self.bucket_paths[bucket])
        digest = self._model_hashes[bucket]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
            raise ValueError(f"Invalid model SHA256 for bucket {bucket}")
        identity = {
            "cache_schema": _CONTEXT_CACHE_SCHEMA,
            "model_sha256": digest.lower(),
            "onnxruntime": _package_version(ort, "onnxruntime"),
            "onnxruntime_qnn": _package_version(qnn, "onnxruntime-qnn"),
            "provider_options": provider_options,
            "context_embed_mode": 1,
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return self.cache_dir / f"backbone_{bucket}_{key}.onnx"

    def _validate_session_inputs(self, session, bucket):
        inputs = {item.name: item for item in session.get_inputs()}
        expected = {
            "inputs_embeds": [1, bucket, self.hidden_size],
            "attn_mask": [1, 1, bucket, bucket],
            "sliding_mask": [1, 1, bucket, bucket],
        }
        if set(inputs) != set(expected):
            raise ValueError(f"Encoder inputs must be {sorted(expected)}, got {sorted(inputs)}")
        for name, shape in expected.items():
            if list(inputs[name].shape) != shape or inputs[name].type != "tensor(float)":
                raise ValueError(
                    f"Manifest bucket {bucket} disagrees with ONNX input {name}: "
                    f"expected float32 {shape}, got {inputs[name].type} {inputs[name].shape}"
                )

    def _new_session(self, ort, model_path, options, bucket):
        session = ort.InferenceSession(str(model_path), sess_options=options)
        # Also disable Python's EP-failure retry, in addition to graph-level
        # session.disable_cpu_ep_fallback, for both source and cached sessions.
        session.disable_fallback()
        self._validate_session_inputs(session, bucket)
        return session

    def _get_session(self, bucket):
        if self._session is not None and self._session_bucket == bucket:
            return self._session
        if self._session is not None:
            self._session = None
            self._session_bucket = None
            self.stats["session_evictions"] += 1
            gc.collect()

        ort, qnn = _qnn_runtime()
        devices = [device for device in ort.get_ep_devices() if device.ep_name == "QNNExecutionProvider"]
        if not devices:
            raise RuntimeError("Qualcomm QNN HTP is unavailable; CPU fallback is disabled")
        provider_options = {
            "backend_path": qnn.get_qnn_htp_path(),
            "htp_arch": "68",
            "htp_performance_mode": "burst",
        }
        if not self.cache_enabled:
            options = self._session_options(ort, devices, provider_options)
            session = self._new_session(ort, self.bucket_paths[bucket], options, bucket)
        else:
            cache_path = self._context_cache_path(bucket, ort, qnn, provider_options)
            self.stats["context_cache_paths"][str(bucket)] = str(cache_path)
            if cache_path.is_file():
                try:
                    options = self._session_options(ort, devices, provider_options, cached_path=cache_path)
                    session = self._new_session(ort, cache_path, options, bucket)
                except Exception as exc:
                    self.stats["context_cache_errors"] += 1
                    raise RuntimeError(
                        f"QNN context cache could not be loaded: {cache_path}. "
                        "Remove this exact cache file to rebuild it, or set LAYA_NPU_CONTEXT_CACHE=0 "
                        "to compile from the source graph. CPU fallback remains disabled."
                    ) from exc
                self.stats["context_cache_hits"] += 1
            else:
                self.stats["context_cache_misses"] += 1
                try:
                    self.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                    with tempfile.TemporaryDirectory(prefix=f".build-{bucket}-", dir=self.cache_dir) as directory:
                        generated_path = Path(directory) / "context.onnx"
                        options = self._session_options(ort, devices, provider_options, generate_path=generated_path)
                        session = self._new_session(ort, self.bucket_paths[bucket], options, bucket)
                        if not generated_path.is_file() or generated_path.stat().st_size == 0:
                            raise RuntimeError("QNN did not generate the requested embedded EPContext model")
                        os.chmod(generated_path, 0o600)
                        with generated_path.open("r+b") as stream:
                            os.fsync(stream.fileno())
                        # Same filesystem, complete file, no external context
                        # paths: readers see either no entry or the whole model.
                        os.replace(generated_path, cache_path)
                    self.stats["context_cache_writes"] += 1
                except Exception as exc:
                    self.stats["context_cache_errors"] += 1
                    raise RuntimeError(
                        f"QNN context generation failed for bucket {bucket} in {self.cache_dir}. "
                        "Check the cache directory permissions/free space, or set LAYA_NPU_CONTEXT_CACHE=0 "
                        "to compile without saving. CPU fallback remains disabled."
                    ) from exc
        self._session = session
        self._session_bucket = bucket
        self.stats["session_creations"] += 1
        return session

    @torch.no_grad()
    def forward(self, input_ids, attention_mask=None, **kwargs):
        if input_ids.ndim != 2 or input_ids.device.type != "cpu":
            raise ValueError("QNNEncoder expects a two-dimensional CPU input_ids tensor")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask and input_ids must have the same shape")
        batch_size, original_length = input_ids.shape
        bucket = select_bucket(self.bucket_paths, original_length)
        output = torch.zeros((batch_size, original_length, self.hidden_size), dtype=torch.float32)
        # Preserve the upstream collated layout. Use one bucket per request so
        # questions of different lengths never require reloading HTP mid-request.
        lengths = []
        for row in range(batch_size):
            valid = attention_mask[row].bool()
            length = int(valid.sum().item())
            if length < 1 or not valid[:length].all() or valid[length:].any():
                raise ValueError("QNNEncoder requires nonempty sequences with contiguous right padding")
            lengths.append(length)

        with self._lock:
            session = self._get_session(bucket)
            for row, length in enumerate(lengths):
                padded = torch.full((1, bucket), self.pad_token_id, dtype=torch.long)
                padded[0, :length] = input_ids[row, :length]
                embeds = self.tok_embeddings(padded).float().numpy()
                global_mask, local_mask = attention_masks(
                    bucket, length, self.local_radius, self.mask_penalty
                )
                result = session.run(None, {
                    "inputs_embeds": embeds,
                    "attn_mask": global_mask,
                    "sliding_mask": local_mask,
                })[0]
                if result.shape != (1, bucket, self.hidden_size) or not np.isfinite(result).all():
                    raise RuntimeError(f"HTP returned invalid hidden states for bucket {bucket}")
                output[row, :length] = torch.from_numpy(np.asarray(result[0, :length], dtype=np.float32))
                self.stats["npu_calls"] += 1
                counts = self.stats["bucket_calls"]
                counts[str(bucket)] = counts.get(str(bucket), 0) + 1
        return SimpleNamespace(last_hidden_state=output)


class NPUAgent:
    """Delegate all Laya behavior to upstream, replacing only its encoder."""

    def __init__(self, model_id_or_path="models/multilingual", manifest_path=None):
        if manifest_path is None:
            manifest_path = os.environ.get(
                "LAYA_NPU_MANIFEST", str(Path(__file__).resolve().parent / "npu/fidelity/manifest.json")
            )
        self.manifest_path = Path(manifest_path).resolve()
        bucket_paths, metadata = load_bucket_manifest(self.manifest_path)
        self.manifest = metadata
        self._fidelity_metadata = artifact_metadata(
            self.manifest_path, metadata, bucket_paths, Path(model_id_or_path) / "model.safetensors"
        )
        self._lock = threading.RLock()

        from laya import load as load_upstream

        self._agent = load_upstream(str(model_id_or_path), device="cpu")
        encoder = self._agent.model.encoder
        adapter = QNNEncoder(
            encoder, bucket_paths, metadata.get("mask_penalty", -100.0),
            model_hashes={int(bucket): info["sha256"] for bucket, info in self._fidelity_metadata["models"].items()},
        )
        adapter.stats.update({
            "max_len": self._agent.cfg.get("max_len", 512),
            "head_max_len": self._agent.cfg.get("head_max_len", 192),
        })
        self._agent.model.encoder = adapter
        self._agent.model.eval()
        # The checkpoint's full encoder was needed for upstream architecture and
        # weight validation; retain only its embedding module after replacement.
        del encoder
        gc.collect()

    def __getattr__(self, name):
        return getattr(self._agent, name)

    @property
    def npu_stats(self):
        return self._agent.model.encoder.stats

    def fidelity_metadata(self):
        """Return cached artifact identities without rehashing large models."""
        return copy.deepcopy(self._fidelity_metadata)

    def fidelity_sequences(self, state, questions):
        """Expose original sequences/markers for independent benchmark checks."""
        from laya.common import build_sequence

        sequences = {}
        for qid, question in questions.items():
            ids, markers = build_sequence(
                self._agent.tok, state, self._agent._to_internal(question),
                self._agent.cfg.get("max_len", 512), self._agent.cfg.get("head_max_len", 192),
            )
            sequences[qid] = {"ids": ids, "markers": markers}
        return sequences

    def predict(self, state: Union[str, dict, list], questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        with self._lock:
            return self._agent.predict(state, questions)

    system_one = predict


def load(model_id_or_path="models/multilingual", device: Optional[str] = "npu", manifest_path=None):
    if device not in (None, "npu"):
        raise ValueError("laya_npu.load requires device='npu'; use laya.load for a CPU agent")
    return NPUAgent(model_id_or_path, manifest_path=manifest_path)
