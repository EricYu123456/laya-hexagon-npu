"""Fold interior ModernBERT LayerNorm affine parameters into projections.

For z = normalize(x), Linear(gamma*z + beta; W, b) is equivalent to
Linear(z; W*gamma, b + W*beta). The mean, variance and epsilon calculation
stays in LayerNorm. Unit LayerNorm weights avoid quantizing learned gamma
with a separate low-precision weight encoding.

Apply before GeGLU grouping and Linear-to-Conv conversion when possible.
Already-grouped GeGLU and ConvLinear projections are also supported.
Embedding normalization and final normalization are deliberately untouched:
their affine outputs participate in residual state or leave the encoder.
"""

from __future__ import annotations

import torch
from torch import nn

from npu.conv_linear import ConvLinear


def _projection_storage(projection):
    if isinstance(projection, nn.Linear):
        return projection, projection.weight
    if isinstance(projection, ConvLinear):
        return projection.conv, projection.conv.weight[:, :, 0, 0]
    raise TypeError("LayerNorm affine folding requires nn.Linear or ConvLinear projections")


def _mlp_inputs(mlp):
    if hasattr(mlp, "Wi"):
        return [("mlp.Wi", mlp.Wi)]
    if hasattr(mlp, "groups"):
        result = []
        for index, group in enumerate(mlp.groups):
            if not hasattr(group, "gate") or not hasattr(group, "value"):
                raise TypeError("Grouped MLP must expose gate and value projections")
            result.extend([(f"mlp.groups.{index}.gate", group.gate),
                           (f"mlp.groups.{index}.value", group.value)])
        if result:
            return result
    raise TypeError("Expected ModernBERT Wi or grouped GeGLU input projections")


@torch.no_grad()
def fold_interior_norms(encoder):
    """Modify interior affine parameters in-place and return build metadata.

    The original parameter objects are retained wherever possible. A projection
    bias is created only when an absent bias must absorb a nonzero norm beta.
    Calling this function again is a no-op. Unsupported shapes/modules fail
    validation before any parameter is changed.
    """
    plans = []
    skipped = []
    seen_weights = set()
    for layer_index, layer in enumerate(encoder.layers):
        entries = [("attn_norm", layer.attn_norm, [("attn.Wqkv", layer.attn.Wqkv)]),
                   ("mlp_norm", layer.mlp_norm, _mlp_inputs(layer.mlp))]
        for norm_name, norm, projections in entries:
            name = f"layers.{layer_index}.{norm_name}"
            if isinstance(norm, nn.Identity):
                skipped.append({"name": name, "reason": "identity"})
                continue
            if not isinstance(norm, nn.LayerNorm) or len(norm.normalized_shape) != 1:
                raise TypeError(f"{name} must be a one-dimensional LayerNorm or Identity")
            if norm.weight is None:
                skipped.append({"name": name, "reason": "no_affine"})
                continue
            gamma = norm.weight.detach().clone()
            beta = norm.bias.detach().clone() if norm.bias is not None else torch.zeros_like(gamma)
            if not torch.isfinite(gamma).all() or not torch.isfinite(beta).all():
                raise ValueError(f"{name} has non-finite affine parameters")
            if bool(torch.all(gamma == 1)) and not bool(torch.any(beta != 0)):
                skipped.append({"name": name, "reason": "already_unit_affine"})
                continue
            checked = []
            for projection_name, projection in projections:
                storage, matrix = _projection_storage(projection)
                if matrix.ndim != 2 or matrix.shape[1] != gamma.numel():
                    raise ValueError(f"{name} and {projection_name} have incompatible feature dimensions")
                if id(storage.weight) in seen_weights:
                    raise ValueError("Cannot fold an input projection shared by multiple normalization paths")
                seen_weights.add(id(storage.weight))
                checked.append((f"layers.{layer_index}.{projection_name}", storage, matrix))
            plans.append((name, norm, gamma, beta, checked))

    changes = []
    for name, norm, gamma, beta, projections in plans:
        changed_projections = []
        for projection_name, storage, matrix in projections:
            gamma_local = gamma.to(device=matrix.device, dtype=matrix.dtype)
            beta_local = beta.to(device=matrix.device, dtype=matrix.dtype)
            # Compute W*beta before changing W. Existing bias contributes once.
            shift = matrix @ beta_local if bool(torch.any(beta_local != 0)) else None
            matrix.mul_(gamma_local[None, :])
            created_bias = shift is not None and storage.bias is None
            if shift is not None:
                if storage.bias is None:
                    storage.bias = nn.Parameter(shift, requires_grad=storage.weight.requires_grad)
                else:
                    storage.bias.add_(shift)
            changed_projections.append({"name": projection_name, "created_bias": created_bias})
        norm.weight.fill_(1)
        if norm.bias is not None:
            norm.bias.zero_()
        changes.append({"name": name, "projections": changed_projections})
    return {
        "folded_norms": len(changes),
        "folded_projections": sum(len(change["projections"]) for change in changes),
        "changes": changes,
        "skipped": skipped,
        "preserved": ["embeddings.norm", "final_norm"],
    }
