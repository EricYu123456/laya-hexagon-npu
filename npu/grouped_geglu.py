"""Exact GeGLU partitioning for backends with per-tensor quantization.

Each group has independent gate/value projections, GELU, multiplication and
output projection. No wide Wi or GeGLU intermediate is shared between groups,
so an outlier channel need not determine other groups' quantization ranges.
This changes the floating-point summation order, but never clips activations
or approximates weights. It is intended for inference with dropout disabled.
"""
from __future__ import annotations

import copy
import math
from typing import Sequence

import torch
from torch import nn


def partition_geglu_channels(
    channel_maxima: torch.Tensor,
    *,
    strategy: str = "outliers",
    outlier_ratio: float = 16.0,
    max_outliers: int = 8,
    topk: int | None = None,
    outlier_group_size: int = 1,
    bucket_ratio: float = 16.0,
) -> tuple[tuple[int, ...], ...]:
    """Partition channels using calibration max(abs(GeGLU output)).

    ``outliers`` isolates at most ``max_outliers`` channels whose maxima are
    at least ``outlier_ratio`` times the median positive maximum. Each is
    isolated by default; ``outlier_group_size`` can combine adjacent selected
    channels. Setting ``topk`` explicitly selects that many largest channels,
    regardless of the ratio, subject to ``max_outliers``.

    ``powers`` groups maxima in geometric buckets relative to the median
    positive maximum: below bucket_ratio, then [ratio, ratio**2), etc. This
    strategy has no top-k cap. All-zero calibration uses one group.

    Every channel occurs exactly once, including zero-range channels. The
    returned indices describe parameter slicing only, not runtime Gather ops.
    """
    if channel_maxima.ndim != 1 or channel_maxima.numel() == 0:
        raise ValueError("channel_maxima must be a nonempty one-dimensional tensor")
    values = channel_maxima.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("channel_maxima must contain finite nonnegative values")
    if strategy not in {"outliers", "powers"}:
        raise ValueError("strategy must be 'outliers' or 'powers'")
    if not math.isfinite(outlier_ratio) or outlier_ratio <= 1:
        raise ValueError("outlier_ratio must be finite and greater than one")
    if not math.isfinite(bucket_ratio) or bucket_ratio <= 1:
        raise ValueError("bucket_ratio must be finite and greater than one")
    if max_outliers < 0 or outlier_group_size < 1:
        raise ValueError("max_outliers must be nonnegative and outlier_group_size positive")
    if topk is not None and (topk < 0 or topk > max_outliers):
        raise ValueError("topk must be between zero and max_outliers")
    if strategy == "powers" and topk is not None:
        raise ValueError("topk applies only to the outliers strategy")

    maxima = values.tolist()
    positive = values[values > 0]
    baseline = float(positive.median()) if positive.numel() else 1.0
    if strategy == "powers":
        buckets: dict[int, list[int]] = {}
        log_ratio = math.log(bucket_ratio)
        for index, value in enumerate(maxima):
            level = max(0, math.floor(math.log(value / baseline) / log_ratio + 1e-12)) if value > 0 else 0
            buckets.setdefault(level, []).append(index)
        return tuple(tuple(buckets[level]) for level in sorted(buckets))

    ranked = sorted(range(len(maxima)), key=lambda index: (-maxima[index], index))
    if topk is None:
        selected = [i for i in ranked if maxima[i] >= baseline * outlier_ratio][:max_outliers]
    else:
        selected = ranked[:topk]
    selected_set = set(selected)
    normal = tuple(i for i in range(len(maxima)) if i not in selected_set)
    groups = [normal] if normal else []
    groups.extend(tuple(selected[i:i + outlier_group_size]) for i in range(0, len(selected), outlier_group_size))
    return tuple(groups)


def _copy_projection(source: nn.Linear, rows: Sequence[int] | None = None,
                     columns: Sequence[int] | None = None, *, include_bias: bool = True) -> nn.Linear:
    weight = source.weight.detach()
    if rows is not None:
        weight = weight[list(rows), :]
    if columns is not None:
        weight = weight[:, list(columns)]
    use_bias = include_bias and source.bias is not None
    result = nn.Linear(weight.shape[1], weight.shape[0], bias=use_bias,
                       device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        result.weight.copy_(weight)
        if use_bias:
            bias = source.bias.detach() if rows is None else source.bias.detach()[list(rows)]
            result.bias.copy_(bias)
            result.bias.requires_grad_(source.bias.requires_grad)
    result.weight.requires_grad_(source.weight.requires_grad)
    return result


class _GeGLUGroup(nn.Module):
    def __init__(self, original: nn.Module, channels: Sequence[int], *, include_output_bias: bool):
        super().__init__()
        width = original.Wo.in_features
        self.gate = _copy_projection(original.Wi, rows=channels)
        self.value = _copy_projection(original.Wi, rows=[i + width for i in channels])
        self.act = copy.deepcopy(original.act)
        self.drop = copy.deepcopy(original.drop)
        self.output = _copy_projection(original.Wo, columns=channels, include_bias=include_output_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.output(self.drop(self.act(self.gate(hidden_states)) * self.value(hidden_states)))


class GroupedGeGLU(nn.Module):
    """Inference replacement for ModernBERT's Wi/act/drop/Wo MLP.

    Construct with explicit index groups, or call ``from_mlp`` with calibrated
    per-channel GeGLU maxima. The source module is never modified or retained.
    Output bias, if present, is included exactly once in the first group.
    """
    def __init__(self, original: nn.Module, channel_groups: Sequence[Sequence[int]]):
        super().__init__()
        if not isinstance(original.Wi, nn.Linear) or not isinstance(original.Wo, nn.Linear):
            raise TypeError("Wi and Wo must be torch.nn.Linear modules")
        width = original.Wo.in_features
        if original.Wi.out_features != 2 * width:
            raise ValueError("Wi must contain one gate half and one value half matching Wo")
        groups = tuple(tuple(group) for group in channel_groups)
        if not groups or any(not group for group in groups):
            raise ValueError("channel groups must be nonempty")
        flat = [i for group in groups for i in group]
        if any(not isinstance(i, int) or isinstance(i, bool) for i in flat) or sorted(flat) != list(range(width)):
            raise ValueError("groups must partition every GeGLU channel exactly once")
        self.channel_groups = groups
        self.groups = nn.ModuleList([
            _GeGLUGroup(original, channels, include_output_bias=(index == 0))
            for index, channels in enumerate(groups)
        ])
        self.dropout_probability = float(getattr(original.drop, "p", 0.0))
        self.train(original.training)

    @classmethod
    def from_mlp(cls, original: nn.Module, channel_maxima: torch.Tensor, **partition_options) -> "GroupedGeGLU":
        if channel_maxima.numel() != original.Wo.in_features:
            raise ValueError("calibration maxima must match Wo input channels")
        return cls(original, partition_geglu_channels(channel_maxima, **partition_options))

    def summary(self) -> dict:
        """Compact build metadata; does not serialize weights or channel arrays."""
        return {"groups": len(self.groups), "group_sizes": [len(g) for g in self.channel_groups]}

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.training and self.dropout_probability:
            raise RuntimeError("GroupedGeGLU requires eval mode when dropout is nonzero")
        result = self.groups[0](hidden_states)
        for group in self.groups[1:]:
            result = result + group(hidden_states)
        return result
