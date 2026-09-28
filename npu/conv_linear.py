"""Exact sequence Linear -> 1x1 Conv2d conversion for QNN weight quantization.

QNN supports output-channel Conv weight quantization (ONNX weight axis 0),
whereas its MatMul implementation uses per-tensor weights. Convert projections
after any GeGLU grouping/balancing and before exporting an encoder to ONNX.
The activation layout remains [batch, sequence, features] outside each module.
"""
from __future__ import annotations

import torch
from torch import nn


class ConvLinear(nn.Module):
    """Copy an nn.Linear into a mathematically equivalent sequence projection.

    Weights retain every original value: [out, in] becomes [out, in, 1, 1].
    The source module is neither changed nor retained. Floating-point kernels
    can use a different reduction order, so equality is numerical, not bitwise.
    """
    def __init__(self, linear: nn.Linear):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("ConvLinear requires an nn.Linear source")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.conv = nn.Conv2d(
            self.in_features, self.out_features, kernel_size=1,
            bias=linear.bias is not None, device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            self.conv.weight.copy_(linear.weight[:, :, None, None])
            if linear.bias is not None:
                self.conv.bias.copy_(linear.bias)
                self.conv.bias.requires_grad_(linear.bias.requires_grad)
        self.conv.weight.requires_grad_(linear.weight.requires_grad)
        self.train(linear.training)

    @property
    def weight(self):
        """A two-dimensional view for introspection of the original projection."""
        return self.conv.weight[:, :, 0, 0]

    @property
    def bias(self):
        return self.conv.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("ConvLinear expects [batch, sequence, features]")
        channels_first = hidden_states.transpose(1, 2).unsqueeze(-1)
        projected = self.conv(channels_first)
        return projected.squeeze(-1).transpose(1, 2).contiguous()


def replace_linears_with_conv(module: nn.Module) -> int:
    """Replace descendant Linear modules in-place; return unique module count.

    This deliberately affects only the supplied subtree: pass the encoder, not
    the whole agent, to leave CPU decision heads unchanged. Shared Linear
    module references remain shared. Repeated calls are idempotent. Convert a
    root Linear directly with ConvLinear rather than passing it to this helper.
    """
    if isinstance(module, nn.Linear):
        raise ValueError("Wrap a root Linear with ConvLinear(linear) directly")
    replacements: dict[int, ConvLinear] = {}
    visited: set[int] = set()

    def visit(parent):
        if id(parent) in visited or isinstance(parent, ConvLinear):
            return
        visited.add(id(parent))
        # named_children() deduplicates aliases; _modules keeps every attribute
        # that must point to the same replacement for tied module references.
        for name, child in tuple(parent._modules.items()):
            if isinstance(child, nn.Linear):
                replacement = replacements.get(id(child))
                if replacement is None:
                    replacement = ConvLinear(child)
                    replacements[id(child)] = replacement
                setattr(parent, name, replacement)
            elif child is not None:
                visit(child)

    visit(module)
    return len(replacements)


def qnn_conv_weight_overrides(model) -> dict:
    """Use QNN's tested U16/S8 per-channel Conv recipe without changing MatMul.

    Merge this dictionary into get_qnn_qdq_config's init_overrides. Keep its
    global weight_type=QUInt8 for other operators, activation_type=QUInt16,
    per_channel=True, and existing attention activation conversion overrides.

    Primary reference: ORT's QnnHTPBackendTests.ConvU16S8S32_PerChannel,
    https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/providers/qnn/conv_test.cc
    The QNN Conv builder requires axis 0 and static weights for per-channel
    quantization. These options specify signed, symmetric INT8 explicitly;
    per_channel=True alone does not select that weight type.
    """
    from onnxruntime.quantization import QuantType

    initializers = {initializer.name for initializer in model.graph.initializer}
    overrides = {}
    for node in model.graph.node:
        if node.op_type == "Conv":
            if len(node.input) < 2 or node.input[1] not in initializers:
                raise ValueError(f"Conv {node.name!r} needs static weights for per-channel quantization")
            overrides[node.input[1]] = [{"quant_type": QuantType.QInt8, "axis": 0, "symmetric": True}]
    return overrides
