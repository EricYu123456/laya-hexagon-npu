"""Experimental feature-separated ModernBERT residual streams.

Output-projection rows and residual additions remain in separate feature bands.
Before each LayerNorm, every band is divided by the SAME per-token maximum
absolute value (clamped to at least one). Only these bounded values are joined.

IMPORTANT: the native LayerNorm uses a fixed small epsilon after this scaling.
Exact equivalence would require original_epsilon / scale**2. This implementation
is therefore an approximation, especially for almost-constant input vectors.
Validate FP32 outputs against the pristine encoder before quantization; a
suggested maximum absolute error gate is 1e-2. QNN support/accuracy is unverified.
"""
from __future__ import annotations

import copy
import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def feature_partition(hidden_size: int, *, outlier_features: Sequence[int] = (488, 580),
                      feature_groups: Sequence[Sequence[int]] | None = None) -> tuple[tuple[int, ...], ...]:
    """Validate a complete disjoint partition; preserve each supplied order."""
    if hidden_size < 1:
        raise ValueError("hidden_size must be positive")
    if feature_groups is None:
        outliers = tuple(outlier_features)
        if len(set(outliers)) != len(outliers):
            raise ValueError("outlier_features must be unique")
        selected = set(outliers)
        feature_groups = [tuple(i for i in range(hidden_size) if i not in selected), outliers]
    groups = tuple(tuple(group) for group in feature_groups)
    flat = [index for group in groups for index in group]
    if (not groups or any(not group for group in groups)
            or any(not isinstance(index, int) or isinstance(index, bool) for index in flat)
            or sorted(flat) != list(range(hidden_size))):
        raise ValueError("feature groups must partition every hidden feature exactly once")
    return groups


@torch.no_grad()
def collect_feature_ranges(encoder, calibration_feeds, *, progress=None) -> dict:
    """Measure original raw inputs of every norm with static padded feed masks.

Each feed supplies inputs_embeds, attn_mask and sliding_mask, as NumPy arrays or
Torch tensors. Every position, including CLS and padding, contributes. Inputs
must already follow the intended padding policy (e.g. zero padding embeddings).
This function uses the original epsilon and does not apply feature scaling. It
supports ordinary or GroupedGeGLU MLPs and Linear/ConvLinear projections.
Only shallow attention copies receive an eager-attention configuration.
"""
    if encoder.training:
        raise ValueError("feature calibration requires an eval encoder")
    parameter = next(encoder.parameters())
    hidden_size = int(encoder.config.hidden_size)
    overall = torch.zeros(hidden_size, device=parameter.device, dtype=torch.float32)
    by_norm = {}
    attention_copies = []
    for layer in encoder.layers:
        attention = copy.copy(layer.attn)
        attention._modules = layer.attn._modules.copy()
        attention.config = copy.copy(layer.attn.config)
        attention.config._attn_implementation = "eager"
        attention.config.reference_compile = False
        attention_copies.append(attention)
    rotary_cache = {}
    lengths = set()
    count = 0

    def observe(name, hidden):
        maxima = hidden.detach().abs().amax(tuple(range(hidden.ndim - 1))).float()
        if not torch.isfinite(maxima).all():
            raise ValueError(f"non-finite feature calibration at {name}")
        overall.copy_(torch.maximum(overall, maxima))
        by_norm[name] = torch.maximum(by_norm[name], maxima) if name in by_norm else maxima.clone()

    for feed in calibration_feeds:
        embeds = torch.as_tensor(feed["inputs_embeds"], device=parameter.device, dtype=parameter.dtype)
        attn_mask = torch.as_tensor(feed["attn_mask"], device=parameter.device, dtype=parameter.dtype)
        sliding_mask = torch.as_tensor(feed["sliding_mask"], device=parameter.device, dtype=parameter.dtype)
        if embeds.ndim != 3 or embeds.shape[-1] != hidden_size:
            raise ValueError("inputs_embeds must have shape [batch, sequence, hidden_size]")
        length = int(embeds.shape[1]); lengths.add(length)
        if length not in rotary_cache:
            rotary_cache[length] = {
                kind: encoder.rotary_emb(embeds, torch.arange(length, device=parameter.device)[None], kind)
                for kind in set(encoder.config.layer_types)
            }
        hidden = encoder.embeddings.norm(embeds)
        for index, (layer, attention) in enumerate(zip(encoder.layers, attention_copies)):
            observe(f"layers.{index}.attn_norm", hidden)
            context = attention(layer.attn_norm(hidden), attention_mask=attn_mask,
                                sliding_window_mask=sliding_mask, position_ids=None,
                                position_embeddings=rotary_cache[length][layer.attention_type])[0]
            hidden = hidden + context
            observe(f"layers.{index}.mlp_norm", hidden)
            hidden = hidden + layer.mlp(layer.mlp_norm(hidden))
        observe("final_norm", hidden)
        count += 1
        if progress is not None:
            progress(count)
    if count == 0:
        raise ValueError("at least one calibration feed is required")
    return {"format_version": 1, "kind": "raw_residual_feature_calibration", "samples": count,
            "hidden_size": hidden_size, "sequence_lengths": sorted(lengths), "includes_cls_and_padding": True,
            "max_abs": overall.cpu().tolist(),
            "by_normalization": {name: values.cpu().tolist() for name, values in by_norm.items()}}


def make_feature_group_plan(calibration: dict, *, thresholds=(100.0, 500.0, 2000.0)) -> dict:
    """Build one common feature partition from reserved calibration maxima.

The JSON-ready schema's `groups` value can be passed directly to
FeatureSplitBackbone(feature_groups=...). Empty bands are omitted. Thresholds
are inclusive upper bounds, followed by an unbounded final band.
"""
    maxima = torch.as_tensor(calibration["max_abs"], dtype=torch.float64)
    limits = tuple(float(value) for value in thresholds)
    if (maxima.ndim != 1 or maxima.numel() == 0 or not torch.isfinite(maxima).all()
            or (maxima < 0).any() or any(not math.isfinite(x) or x <= 0 for x in limits)
            or tuple(sorted(set(limits))) != limits):
        raise ValueError("maxima and strictly increasing positive thresholds are required")
    groups = [[] for _ in range(len(limits) + 1)]
    for index, value in enumerate(maxima.tolist()):
        band = sum(value > limit for limit in limits)
        groups[band].append(index)
    groups = [group for group in groups if group]
    feature_partition(len(maxima), feature_groups=groups)
    return {"format_version": 1, "kind": "calibrated_feature_groups", "groups": groups,
            "thresholds": list(limits), "group_sizes": [len(group) for group in groups],
            "group_max_abs": [float(maxima[group].max()) for group in groups],
            "max_abs": maxima.tolist(), "calibration_samples": calibration["samples"],
            "sequence_lengths": calibration["sequence_lengths"],
            "includes_cls_and_padding": bool(calibration["includes_cls_and_padding"])}


class SplitOutputProjection(nn.Module):
    """Project directly into feature bands, without materializing a wide output.

Supports nn.Linear and this project's ConvLinear. ConvLinear sources retain
ConvLinear branches; Linear branches can be converted afterwards by the normal
replace_linears_with_conv helper. The source projection is never modified.
"""
    def __init__(self, source: nn.Module, groups: Sequence[Sequence[int]]):
        super().__init__()
        from npu.conv_linear import ConvLinear

        if not isinstance(source, (nn.Linear, ConvLinear)):
            raise TypeError("SplitOutputProjection needs Linear or ConvLinear")
        self.groups = feature_partition(source.out_features, feature_groups=groups)
        branches = []
        for group in self.groups:
            branch = nn.Linear(source.in_features, len(group), bias=source.bias is not None,
                               device=source.weight.device, dtype=source.weight.dtype)
            with torch.no_grad():
                branch.weight.copy_(source.weight[list(group)])
                if source.bias is not None:
                    branch.bias.copy_(source.bias[list(group)])
                    branch.bias.requires_grad_(source.bias.requires_grad)
            branch.weight.requires_grad_(source.weight.requires_grad)
            branch.train(source.training)
            branches.append(ConvLinear(branch) if isinstance(source, ConvLinear) else branch)
        self.branches = nn.ModuleList(branches)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(branch(hidden_states) for branch in self.branches)


class ScaledFeatureLayerNorm(nn.Module):
    """Join only bounded residual bands, normalize, and restore feature order.

For an Identity source (the first attention norm), reconstruct without scaling.
There is no scale invariance to exploit for Identity, and the initial embedding
norm has not yet developed the large residual outliers this wrapper targets.

collect_stats is an eager-only diagnostic. It compares this norm with original
epsilon on the SAME unscaled streams; it does not measure encoder-wide error.
Statistics are excluded from tracing/export and do not change model outputs.
"""
    def __init__(self, source: nn.Module, groups: Sequence[Sequence[int]], *,
                 epsilon: float = 1e-12, collect_stats: bool = False):
        super().__init__()
        if not isinstance(source, (nn.LayerNorm, nn.Identity)):
            raise TypeError("ScaledFeatureLayerNorm supports LayerNorm or Identity")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.groups = tuple(tuple(group) for group in groups)
        hidden_size = sum(map(len, self.groups))
        feature_partition(hidden_size, feature_groups=self.groups)
        permutation = torch.tensor([i for group in self.groups for i in group], dtype=torch.long)
        self.register_buffer("inverse_permutation", torch.argsort(permutation))
        self.hidden_size = hidden_size
        self.is_identity = isinstance(source, nn.Identity)
        self.collect_stats = collect_stats
        self.last_stats: dict | None = None
        self.original_epsilon = None if self.is_identity else float(source.eps)
        if self.is_identity:
            self.norm = nn.Identity()
        else:
            if tuple(source.normalized_shape) != (hidden_size,):
                raise ValueError("source LayerNorm must normalize the full hidden feature axis")
            parameter = source.weight if source.weight is not None else source.bias
            factory = {} if parameter is None else {"device": parameter.device, "dtype": parameter.dtype}
            self.norm = nn.LayerNorm(hidden_size, eps=epsilon,
                                     elementwise_affine=source.elementwise_affine,
                                     bias=source.bias is not None, **factory)
            with torch.no_grad():
                if source.weight is not None:
                    self.norm.weight.copy_(source.weight[permutation.to(source.weight.device)])
                    self.norm.weight.requires_grad_(source.weight.requires_grad)
                if source.bias is not None:
                    self.norm.bias.copy_(source.bias[permutation.to(source.bias.device)])
                    self.norm.bias.requires_grad_(source.bias.requires_grad)
            self.inverse_permutation = self.inverse_permutation.to(parameter.device) if parameter is not None else self.inverse_permutation

    def forward(self, streams: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(streams) != len(self.groups):
            raise ValueError("stream count does not match feature partition")
        if self.is_identity:
            return torch.cat(streams, dim=-1).index_select(-1, self.inverse_permutation)
        scale = torch.ones_like(streams[0][..., :1])
        for stream in streams:
            scale = torch.maximum(scale, stream.abs().amax(dim=-1, keepdim=True))
        bounded = torch.cat(tuple(stream / scale for stream in streams), dim=-1)
        normalized = self.norm(bounded)
        if self.collect_stats and not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export():
            with torch.no_grad():
                unscaled = torch.cat(streams, dim=-1)
                original = F.layer_norm(unscaled, (self.hidden_size,),
                                        self.norm.weight, self.norm.bias, self.original_epsilon)
                difference = (normalized - original).abs()
                variance = unscaled.double().var(dim=-1, keepdim=True, unbiased=False)
                factor_error = ((variance + self.original_epsilon)
                                / (variance + self.norm.eps * scale.double().square())).sqrt().sub(1).abs()
                # A exactly constant token has zero centered input, so even a
                # large denominator ratio cannot change its normalized value.
                nonconstant_factor_error = factor_error[variance > 0]
                self.last_stats = {
                    "original_epsilon": self.original_epsilon,
                    "scaled_epsilon": float(self.norm.eps),
                    "scale_min": float(scale.min()), "scale_max": float(scale.max()),
                    "bounded_input_max_abs": float(bounded.abs().max()),
                    "original_variance_min": float(variance.min()),
                    "original_variance_max": float(variance.max()),
                    "nonconstant_factor_error_max": float(nonconstant_factor_error.max()) if nonconstant_factor_error.numel() else 0.0,
                    "same_streams_mean_abs_error": float(difference.mean()),
                    "same_streams_max_abs_error": float(difference.max()),
                }
        return normalized.index_select(-1, self.inverse_permutation)


class _SplitGeGLUGroup(nn.Module):
    def __init__(self, source, groups):
        super().__init__()
        self.gate = source.gate
        self.value = source.value
        self.act = source.act
        self.drop = source.drop
        self.output = SplitOutputProjection(source.output, groups)

    def forward(self, hidden_states):
        value = self.drop(self.act(self.gate(hidden_states)) * self.value(hidden_states))
        return self.output(value)


class SplitOutputMLP(nn.Module):
    """Compute each GeGLU activation once and split only output rows."""
    def __init__(self, source, groups):
        super().__init__()
        self.grouped = hasattr(source, "groups")
        if self.grouped:
            self.groups = nn.ModuleList([_SplitGeGLUGroup(group, groups) for group in source.groups])
        else:
            self.Wi, self.act, self.drop = source.Wi, source.act, source.drop
            self.Wo = SplitOutputProjection(source.Wo, groups)

    def forward(self, hidden_states):
        if not self.grouped:
            first, second = self.Wi(hidden_states).chunk(2, dim=-1)
            return self.Wo(self.drop(self.act(first) * second))
        result = self.groups[0](hidden_states)
        for group in self.groups[1:]:
            addition = group(hidden_states)
            result = tuple(a + b for a, b in zip(result, addition))
        return result


class _FeatureSplitLayer(nn.Module):
    def __init__(self, source, groups, *, epsilon, collect_stats):
        super().__init__()
        self.attention_type = source.attention_type
        self.attn_norm = ScaledFeatureLayerNorm(source.attn_norm, groups, epsilon=epsilon, collect_stats=collect_stats)
        self.mlp_norm = ScaledFeatureLayerNorm(source.mlp_norm, groups, epsilon=epsilon, collect_stats=collect_stats)
        # Shallow-copy the attention container and its module registry so replacing
        # Wo/out_drop never mutates the pristine encoder's attention module.
        attention = copy.copy(source.attn)
        attention._modules = source.attn._modules.copy()
        attention.config = copy.copy(source.attn.config)
        attention.config._attn_implementation = "eager"
        attention.config.reference_compile = False
        attention.Wo = nn.Identity()
        if hasattr(attention, "out_drop"):
            attention.out_drop = nn.Identity()
        self.attention_context = attention
        self.attention_output = SplitOutputProjection(source.attn.Wo, groups)
        self.mlp = SplitOutputMLP(source.mlp, groups)

    def forward(self, streams, attn_mask, sliding_mask, position_embeddings):
        context = self.attention_context(
            self.attn_norm(streams), attention_mask=attn_mask,
            sliding_window_mask=sliding_mask, position_ids=None,
            position_embeddings=position_embeddings,
        )[0]
        attention = self.attention_output(context)
        streams = tuple(a + b for a, b in zip(streams, attention))
        mlp = self.mlp(self.mlp_norm(streams))
        return tuple(a + b for a, b in zip(streams, mlp))

    def forward_split(self, cls, rest, attn_mask, sliding_mask, position_embeddings):
        normalized = torch.cat((self.attn_norm(cls), self.attn_norm(rest)), dim=1)
        context = self.attention_context(
            normalized, attention_mask=attn_mask,
            sliding_window_mask=sliding_mask, position_ids=None,
            position_embeddings=position_embeddings,
        )[0]
        attention = self.attention_output(context)
        cls = tuple(a + b[:, :1] for a, b in zip(cls, attention))
        rest = tuple(a + b[:, 1:] for a, b in zip(rest, attention))
        cls_mlp = self.mlp(self.mlp_norm(cls))
        rest_mlp = self.mlp(self.mlp_norm(rest))
        return (tuple(a + b for a, b in zip(cls, cls_mlp)),
                tuple(a + b for a, b in zip(rest, rest_mlp)))


class FeatureSplitBackbone(nn.Module):
    """Drop-in three-input export wrapper; does not modify the source encoder.

Use after any norm-affine folding or GroupedGeGLU replacement. Projection
conversion to ConvLinear may happen before OR after constructing this wrapper.
Only inference with dropout disabled is supported. Optional CLS splitting keeps
the activation sink separate within every feature band.
"""
    def __init__(self, encoder, length: int, *, outlier_features=(488, 580),
                 feature_groups=None, epsilon=1e-12, collect_stats=False, split_cls=False):
        super().__init__()
        if length < 1:
            raise ValueError("length must be positive")
        if split_cls and length < 2:
            raise ValueError("CLS splitting requires at least two token positions")
        self.split_cls = bool(split_cls)
        hidden_size = encoder.config.hidden_size
        self.feature_groups = feature_partition(hidden_size, outlier_features=outlier_features,
                                                feature_groups=feature_groups)
        self.embeddings_norm = encoder.embeddings.norm
        for i, group in enumerate(self.feature_groups):
            self.register_buffer(f"feature_indices_{i}", torch.tensor(group, dtype=torch.long))
        self.layers = nn.ModuleList([
            _FeatureSplitLayer(layer, self.feature_groups, epsilon=epsilon, collect_stats=collect_stats)
            for layer in encoder.layers
        ])
        self.final_norm = ScaledFeatureLayerNorm(encoder.final_norm, self.feature_groups,
                                                epsilon=epsilon, collect_stats=collect_stats)
        parameter = next(encoder.parameters())
        for kind in set(encoder.config.layer_types):
            positions = torch.arange(length, device=parameter.device)[None]
            cos, sin = encoder.rotary_emb(torch.zeros(1, length, hidden_size, device=parameter.device), positions, kind)
            self.register_buffer("cos_" + kind, cos)
            self.register_buffer("sin_" + kind, sin)
        self.metadata = {
            "feature_groups": [list(group) for group in self.feature_groups],
            "feature_group_sizes": [len(group) for group in self.feature_groups],
            "split_cls": self.split_cls,
            "scaled_layernorm_epsilon": float(epsilon),
            "approximation": "fixed scaled LayerNorm epsilon replaces original epsilon / per-token scale squared",
            "suggested_fp32_max_abs_error_gate": 0.01,
            "hardware_verified": False,
        }
        self.to(device=parameter.device)
        self.eval()

    def normalization_stats(self):
        return {name: module.last_stats for name, module in self.named_modules()
                if isinstance(module, ScaledFeatureLayerNorm) and module.last_stats is not None}

    def forward(self, inputs_embeds, attn_mask, sliding_mask):
        if self.training:
            raise RuntimeError("FeatureSplitBackbone is an inference-only export wrapper; call eval()")
        hidden = self.embeddings_norm(inputs_embeds)
        streams = tuple(hidden.index_select(-1, getattr(self, f"feature_indices_{i}"))
                        for i in range(len(self.feature_groups)))
        if self.split_cls:
            cls = tuple(stream[:, :1] for stream in streams)
            rest = tuple(stream[:, 1:] for stream in streams)
            for layer in self.layers:
                cls, rest = layer.forward_split(cls, rest, attn_mask, sliding_mask,
                                               (getattr(self, "cos_" + layer.attention_type),
                                                getattr(self, "sin_" + layer.attention_type)))
            return torch.cat((self.final_norm(cls), self.final_norm(rest)), dim=1)
        for layer in self.layers:
            streams = layer(streams, attn_mask, sliding_mask,
                            (getattr(self, "cos_" + layer.attention_type), getattr(self, "sin_" + layer.attention_type)))
        return self.final_norm(streams)
