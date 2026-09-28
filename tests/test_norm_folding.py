"""Cheap functional checks for exact LayerNorm affine folding."""
import copy
import unittest

import torch
from torch import nn

from npu.conv_linear import replace_linears_with_conv
from npu.grouped_geglu import GroupedGeGLU
from npu.norm_folding import fold_interior_norms


class TinyMLP(nn.Module):
    def __init__(self, width, bias):
        super().__init__()
        self.Wi = nn.Linear(width, width * 4, bias=bias)
        self.Wo = nn.Linear(width * 2, width, bias=bias)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0)

    def forward(self, x):
        gate, value = self.Wi(x).chunk(2, dim=-1)
        return self.Wo(self.act(gate) * value)


def make_encoder(norm_bias=True, projection_bias=True):
    width = 8
    encoder = nn.Module()
    encoder.embeddings = nn.Module()
    encoder.embeddings.norm = nn.LayerNorm(width, bias=norm_bias)
    encoder.final_norm = nn.LayerNorm(width, bias=norm_bias)
    encoder.layers = nn.ModuleList()
    for index in range(2):
        layer = nn.Module()
        layer.attn_norm = nn.Identity() if index == 0 else nn.LayerNorm(width, bias=norm_bias)
        layer.attn = nn.Module()
        layer.attn.Wqkv = nn.Linear(width, width * 3, bias=projection_bias)
        layer.mlp_norm = nn.LayerNorm(width, bias=norm_bias)
        layer.mlp = TinyMLP(width, projection_bias)
        encoder.layers.append(layer)
    with torch.no_grad():
        for module in encoder.modules():
            if isinstance(module, nn.LayerNorm):
                module.weight.copy_(torch.randn(width) * 0.7 + 1)
                if module.bias is not None:
                    module.bias.copy_(torch.randn(width) * 0.4)
    return encoder.eval()


class NormFoldingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def check_equivalent(self, encoder):
        before = copy.deepcopy(encoder)
        x = torch.randn(2, 5, 8)
        report = fold_interior_norms(encoder)
        self.assertEqual(report["folded_norms"], 3)
        for original, changed in zip(before.layers, encoder.layers):
            torch.testing.assert_close(original.attn.Wqkv(original.attn_norm(x)),
                                       changed.attn.Wqkv(changed.attn_norm(x)), atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(original.mlp(original.mlp_norm(x)),
                                       changed.mlp(changed.mlp_norm(x)), atol=2e-6, rtol=2e-5)
        for original, changed in ((before.embeddings.norm, encoder.embeddings.norm),
                                  (before.final_norm, encoder.final_norm)):
            for key, value in original.state_dict().items():
                self.assertTrue(torch.equal(value, changed.state_dict()[key]))
        after_once = copy.deepcopy(encoder.state_dict())
        self.assertEqual(fold_interior_norms(encoder)["folded_norms"], 0)
        for key, value in encoder.state_dict().items():
            self.assertTrue(torch.equal(value, after_once[key]))
        return report

    def test_original_bias_free_checkpoint_path(self):
        report = self.check_equivalent(make_encoder(norm_bias=False, projection_bias=False))
        self.assertEqual(report["folded_projections"], 3)
        self.assertFalse(any(p["created_bias"] for item in report["changes"] for p in item["projections"]))

    def test_existing_projection_bias_is_added_once(self):
        self.check_equivalent(make_encoder())

    def test_nonzero_norm_beta_creates_projection_bias(self):
        report = self.check_equivalent(make_encoder(norm_bias=True, projection_bias=False))
        self.assertTrue(all(p["created_bias"] for item in report["changes"] for p in item["projections"]))

    def test_conv_and_grouped_projection_path(self):
        encoder = make_encoder(norm_bias=True, projection_bias=False)
        for layer in encoder.layers:
            layer.mlp = GroupedGeGLU(layer.mlp, [list(range(15)), [15]])
        replace_linears_with_conv(encoder)
        report = self.check_equivalent(encoder)
        self.assertEqual(report["folded_projections"], 9)

    def test_invalid_projection_fails_before_any_mutation(self):
        encoder = make_encoder()
        encoder.layers[1].attn.Wqkv = nn.Linear(7, 24)
        before = copy.deepcopy(encoder.state_dict())
        with self.assertRaises(ValueError):
            fold_interior_norms(encoder)
        for key, value in encoder.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))


if __name__ == "__main__":
    unittest.main()
