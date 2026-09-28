"""CPU equivalence and partition correctness: python -m unittest npu.test_grouped_geglu -v."""
import unittest

import torch
from torch import nn

from npu.grouped_geglu import GroupedGeGLU, partition_geglu_channels


class ReferenceGeGLU(nn.Module):
    def __init__(self, hidden=48, intermediate=73, *, bias=False, approximate="none", dropout=0.0):
        super().__init__()
        self.Wi = nn.Linear(hidden, 2 * intermediate, bias=bias)
        self.Wo = nn.Linear(intermediate, hidden, bias=bias)
        self.act = nn.GELU(approximate=approximate)
        self.drop = nn.Dropout(dropout)

    def forward(self, hidden_states):
        gate, value = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(gate) * value))


class GroupedGeGLUTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_outputs_match_original_without_mutating_weights(self):
        for bias in (False, True):
            for approximate in ("none", "tanh"):
                with self.subTest(bias=bias, approximate=approximate):
                    torch.manual_seed(71)
                    original = ReferenceGeGLU(bias=bias, approximate=approximate, dropout=0.2).eval()
                    if bias:
                        with torch.no_grad():
                            original.Wo.bias.copy_(torch.linspace(-7, 5, 48))
                    before = {name: value.clone() for name, value in original.state_dict().items()}
                    maxima = torch.ones(73)
                    maxima[[2, 13, 47]] = torch.tensor([80.0, 10000.0, 1500.0])
                    grouped = GroupedGeGLU.from_mlp(original, maxima)
                    self.assertEqual(grouped.summary(), {"groups": 4, "group_sizes": [70, 1, 1, 1]})
                    for shape, scale in (((1, 1, 48), 1), ((2, 129, 48), 3), ((1, 768, 48), 1)):
                        inputs = torch.randn(*shape) * scale
                        with torch.no_grad():
                            error = (grouped(inputs) - original(inputs)).abs().max().item()
                        self.assertLess(error, 1e-4, f"shape={shape}, max_abs_error={error}")
                    for name, value in original.state_dict().items():
                        torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_group_parameters_are_exact_and_have_separate_projections(self):
        torch.manual_seed(72)
        original = ReferenceGeGLU(hidden=12, intermediate=9, bias=True).eval()
        partition = ((0, 2, 3, 5, 7), (8, 1), (6, 4))
        grouped = GroupedGeGLU(original, partition)
        self.assertFalse(hasattr(grouped, "Wi"))
        for index, (channels, branch) in enumerate(zip(partition, grouped.groups)):
            torch.testing.assert_close(branch.gate.weight, original.Wi.weight[list(channels)], rtol=0, atol=0)
            torch.testing.assert_close(branch.value.weight, original.Wi.weight[[i + 9 for i in channels]], rtol=0, atol=0)
            torch.testing.assert_close(branch.output.weight, original.Wo.weight[:, list(channels)], rtol=0, atol=0)
            self.assertEqual(branch.gate.out_features, len(channels))
            self.assertEqual(branch.value.out_features, len(channels))
            self.assertEqual(branch.output.bias is not None, index == 0)
            self.assertNotEqual(branch.gate.weight.data_ptr(), original.Wi.weight.data_ptr())
        inputs = torch.randn(2, 19, 12)
        torch.testing.assert_close(grouped(inputs), original(inputs), rtol=0, atol=1e-4)

    def test_ratio_topk_and_power_buckets(self):
        maxima = torch.tensor([1.0] * 20 + [16.0, 255.0, 256.0, 4096.0])
        threshold = partition_geglu_channels(maxima, max_outliers=2)
        self.assertEqual(threshold[1:], ((23,), (22,)))
        forced = partition_geglu_channels(torch.ones(5), topk=3, max_outliers=3, outlier_group_size=2)
        self.assertEqual(forced, ((3, 4), (0, 1), (2,)))
        powers = partition_geglu_channels(maxima, strategy="powers")
        self.assertEqual(powers, (tuple(range(20)), (20, 21), (22,), (23,)))
        torch.manual_seed(73)
        original = ReferenceGeGLU(hidden=16, intermediate=24).eval()
        grouped = GroupedGeGLU.from_mlp(original, maxima, strategy="powers")
        inputs = torch.randn(1, 128, 16)
        torch.testing.assert_close(grouped(inputs), original(inputs), rtol=0, atol=1e-4)

    def test_zero_ranges_and_single_group_still_preserve_function(self):
        for maxima in (torch.zeros(9), torch.ones(9)):
            self.assertEqual(partition_geglu_channels(maxima), (tuple(range(9)),))
        original = ReferenceGeGLU(hidden=12, intermediate=9).eval()
        grouped = GroupedGeGLU.from_mlp(original, torch.zeros(9))
        inputs = torch.randn(2, 17, 12)
        torch.testing.assert_close(grouped(inputs), original(inputs), rtol=0, atol=1e-4)

    def test_rejects_incomplete_or_duplicate_channel_partitions(self):
        original = ReferenceGeGLU(hidden=4, intermediate=3).eval()
        for groups in (((0, 1),), ((0, 1), (1, 2)), ((0, 1, 2), ()), ((0, 1, 3),)):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                GroupedGeGLU(original, groups)
        for maxima in (torch.tensor([1.0, float("nan")]), torch.tensor([-1.0, 0.0]), torch.ones(2, 2)):
            with self.assertRaises(ValueError):
                partition_geglu_channels(maxima)

    def test_stochastic_training_requires_explicit_eval(self):
        original = ReferenceGeGLU(dropout=0.5)
        grouped = GroupedGeGLU.from_mlp(original, torch.ones(73), topk=2)
        inputs = torch.randn(1, 4, 48)
        with self.assertRaisesRegex(RuntimeError, "eval mode"):
            grouped(inputs)
        grouped.eval()
        original.eval()
        torch.testing.assert_close(grouped(inputs), original(inputs), rtol=0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
