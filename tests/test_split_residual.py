"""Cheap numerical tests for experimental feature-separated residual export."""
import unittest

import torch
from torch import nn
from types import SimpleNamespace

from npu.conv_linear import ConvLinear, replace_linears_with_conv
from npu.grouped_geglu import GroupedGeGLU
from npu.split_residual import (FeatureSplitBackbone, ScaledFeatureLayerNorm,
                                SplitOutputMLP, SplitOutputProjection, collect_feature_ranges,
                                feature_partition, make_feature_group_plan)


def restore(streams, groups):
    permutation = torch.tensor([i for group in groups for i in group])
    return torch.cat(streams, -1).index_select(-1, torch.argsort(permutation))


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wi = nn.Linear(8, 24)
        self.Wo = nn.Linear(12, 8)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0)

    def forward(self, x):
        a, b = self.Wi(x).chunk(2, -1)
        return self.Wo(self.drop(self.act(a) * b))


class SplitResidualTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        torch.set_num_threads(1)
        self.groups = ((0, 2, 3, 4, 5, 7), (6, 1))

    def test_partition_validation(self):
        self.assertEqual(feature_partition(4, outlier_features=(3, 1)), ((0, 2), (3, 1)))
        for groups in [((0, 1), (1, 2, 3)), ((0, 1), (2,)), ((), (0, 1, 2, 3))]:
            with self.assertRaises(ValueError):
                feature_partition(4, feature_groups=groups)

    def test_calibration_includes_cls_padding_and_interior_residuals(self):
        class ZeroAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace()

            def forward(self, hidden, **kwargs):
                return (torch.zeros_like(hidden),)

        class ConstantMLP(nn.Module):
            def forward(self, hidden):
                return torch.tensor([50., 0., 0.]).expand_as(hidden)

        encoder = nn.Module()
        encoder.anchor = nn.Parameter(torch.zeros(1))
        encoder.config = SimpleNamespace(hidden_size=3, layer_types=["global_attention"])
        encoder.embeddings = nn.Module()
        encoder.embeddings.norm = nn.Identity()
        layer = nn.Module()
        layer.attn_norm, layer.mlp_norm = nn.Identity(), nn.Identity()
        layer.attn, layer.mlp = ZeroAttention(), ConstantMLP()
        layer.attention_type = "global_attention"
        encoder.layers = nn.ModuleList([layer])
        encoder.rotary_emb = lambda embeds, positions, kind: (torch.zeros_like(embeds), torch.zeros_like(embeds))
        encoder.eval()
        # Large first/last positions stand for CLS and padding; both must count.
        x = torch.tensor([[[0., 500., 0.], [70., 0., 0.], [0., 0., 2500.]]])
        feed = {"inputs_embeds": x, "attn_mask": torch.zeros(1, 1, 3, 3), "sliding_mask": torch.zeros(1, 1, 3, 3)}
        ranges = collect_feature_ranges(encoder, [feed])
        self.assertEqual(ranges["max_abs"], [120., 500., 2500.])
        self.assertEqual(ranges["by_normalization"]["layers.0.attn_norm"], [70., 500., 2500.])
        self.assertTrue(ranges["includes_cls_and_padding"])
        plan = make_feature_group_plan(ranges)
        self.assertEqual(plan["groups"], [[0, 1], [2]])
        self.assertEqual(plan["group_max_abs"], [500., 2500.])
        for limits in ((500, 100), (100, 100), (0, 500), (float("nan"),)):
            with self.assertRaises(ValueError):
                make_feature_group_plan(ranges, thresholds=limits)

    def test_output_rows_preserve_bias_and_both_conv_conversion_orders(self):
        source = nn.Linear(5, 8).eval()
        x = torch.randn(2, 7, 5)
        for projection in (source, ConvLinear(source)):
            split = SplitOutputProjection(projection, self.groups)
            torch.testing.assert_close(restore(split(x), self.groups), source(x), atol=1e-6, rtol=1e-6)
            replace_linears_with_conv(split)
            torch.testing.assert_close(restore(split(x), self.groups), source(x), atol=1e-6, rtol=1e-6)

    def test_grouped_and_plain_mlp_share_gate_value_computation(self):
        plain = TinyMLP().eval()
        grouped = GroupedGeGLU(plain, ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (10,), (11,))).eval()
        x = torch.randn(2, 7, 8)
        for source in (plain, grouped):
            split = SplitOutputMLP(source, self.groups).eval()
            calls = []
            modules = [group.gate for group in split.groups] if split.grouped else [split.Wi]
            handles = [module.register_forward_hook(lambda *_: calls.append(1)) for module in modules]
            actual = restore(split(x), self.groups)
            for handle in handles:
                handle.remove()
            self.assertEqual(len(calls), len(modules))
            torch.testing.assert_close(actual, source(x), atol=1e-6, rtol=1e-6)

    def test_identity_never_rescales_and_zero_layernorm_is_finite(self):
        x = torch.randn(2, 4, 8) * 10000
        streams = tuple(x[..., list(group)] for group in self.groups)
        identity = ScaledFeatureLayerNorm(nn.Identity(), self.groups)
        self.assertTrue(torch.equal(identity(streams), x))
        norm = nn.LayerNorm(8)
        with torch.no_grad():
            norm.bias.copy_(torch.arange(8) / 10)
        scaled = ScaledFeatureLayerNorm(norm, self.groups, collect_stats=True)
        zeros = tuple(torch.zeros_like(stream) for stream in streams)
        y = scaled(zeros)
        self.assertTrue(torch.isfinite(y).all())
        torch.testing.assert_close(y, norm(torch.zeros_like(x)))

    def test_affine_reordering_and_epsilon_approximation_are_explicit(self):
        source = nn.LayerNorm(8, eps=1e-5)
        with torch.no_grad():
            source.weight.copy_(torch.linspace(.3, 1.4, 8))
            source.bias.copy_(torch.linspace(-.2, .6, 8))
        x = torch.randn(2, 4, 8) * 10
        x[:, 0, 1] = 10000
        scaled = ScaledFeatureLayerNorm(source, self.groups, collect_stats=True)
        actual = scaled(tuple(x[..., list(group)] for group in self.groups))
        torch.testing.assert_close(actual, source(x), atol=2e-5, rtol=2e-5)
        self.assertLessEqual(scaled.last_stats['bounded_input_max_abs'], 1)
        # Almost-constant vectors are a genuine approximation limitation.
        near_constant = torch.ones_like(x)
        near_constant[..., 1] += 1e-5
        actual = scaled(tuple(near_constant[..., list(group)] for group in self.groups))
        self.assertTrue(torch.isfinite(actual).all())
        self.assertGreater(float((actual - source(near_constant)).abs().max()), .1)

    def test_tiny_modernbert_wrapper_and_source_are_preserved(self):
        from transformers import ModernBertConfig, ModernBertModel

        config = ModernBertConfig(vocab_size=32, pad_token_id=0, bos_token_id=1, eos_token_id=2,
                                 hidden_size=8, intermediate_size=12,
                                 num_hidden_layers=2, num_attention_heads=2,
                                 max_position_embeddings=32, local_attention=4,
                                 global_attn_every_n_layers=2, reference_compile=False,
                                 attention_dropout=0, embedding_dropout=0, mlp_dropout=0)
        config._attn_implementation = 'eager'
        encoder = ModernBertModel(config).eval()
        original_wo = encoder.layers[0].attn.Wo
        wrapper = FeatureSplitBackbone(encoder, 8, feature_groups=self.groups, collect_stats=True)
        self.assertIs(encoder.layers[0].attn.Wo, original_wo)
        self.assertIsNot(wrapper.layers[0].attention_context, encoder.layers[0].attn)
        x = torch.randn(1, 8, 8)
        x[:, 6:] = 0
        valid = torch.arange(8)[None] < 6
        positions = torch.arange(8)
        global_mask = (positions[None, :] >= 6).expand(8, 8)
        local_mask = global_mask | ((positions[:, None] - positions[None, :]).abs() > 2)
        masks = [mask.float()[None, None] * -100 for mask in (global_mask, local_mask)]
        with torch.no_grad():
            expected = encoder(inputs_embeds=x, attention_mask=valid).last_hidden_state[:, :6]
            actual = wrapper(x, *masks)[:, :6]
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        self.assertTrue(wrapper.normalization_stats())
        wrapper.train()
        with self.assertRaises(RuntimeError):
            wrapper(x, *masks)


if __name__ == '__main__':
    unittest.main()
