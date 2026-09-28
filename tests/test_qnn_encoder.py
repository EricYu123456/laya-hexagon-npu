"""Exercise batching and failure behavior without loading a model or QNN."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from laya_npu import QNNEncoder


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(local_attention=128, pad_token_id=0, hidden_size=4)
        self.embedding = torch.nn.Embedding(16, 4)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(64).reshape(16, 4))

    def get_input_embeddings(self):
        return self.embedding


class EncoderContractTests(unittest.TestCase):
    def test_one_bucket_preserves_original_batch_positions(self):
        encoder = QNNEncoder(TinyEncoder(), {2: "small.onnx", 8: "large.onnx"})
        feeds = []

        def infer(_, feed):
            feeds.append(feed)
            return [feed["inputs_embeds"] * 2]

        encoder._get_session = Mock(return_value=SimpleNamespace(run=infer))
        ids = torch.tensor([[11, 12, 0, 0], [7, 8, 9, 10]])
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
        hidden = encoder(ids, mask).last_hidden_state
        self.assertEqual(tuple(hidden.shape), (2, 4, 4))
        encoder._get_session.assert_called_once_with(8)
        torch.testing.assert_close(hidden[0, :2], encoder.tok_embeddings(ids[0, :2]) * 2)
        torch.testing.assert_close(hidden[1], encoder.tok_embeddings(ids[1]) * 2)
        self.assertTrue(torch.all(hidden[0, 2:] == 0))
        self.assertTrue(np.all(feeds[0]["attn_mask"][..., 2:] == -100))
        self.assertEqual(encoder.stats["npu_calls"], 2)
        self.assertEqual(encoder.stats["cpu_fallbacks"], 0)

    def test_provider_failure_propagates_without_fallback(self):
        encoder = QNNEncoder(TinyEncoder(), {8: "model.onnx"})
        session = SimpleNamespace(run=Mock(side_effect=RuntimeError("HTP graph failed")))
        encoder._get_session = Mock(return_value=session)
        with self.assertRaisesRegex(RuntimeError, "HTP graph failed"):
            encoder(torch.tensor([[1, 2]]))
        self.assertEqual(encoder.stats["npu_calls"], 0)
        self.assertEqual(encoder.stats["cpu_fallbacks"], 0)

    def test_optional_zero_padding_preserves_all_real_embeddings(self):
        encoder = QNNEncoder(TinyEncoder(), {8: "model.onnx"}, zero_pad_embeddings=True)
        feeds = []

        def infer(_, feed):
            feeds.append(feed)
            return [feed["inputs_embeds"]]

        encoder._get_session = Mock(return_value=SimpleNamespace(run=infer))
        # Token ID zero can be a real token; the attention mask determines padding.
        ids = torch.tensor([[0, 3, 0], [5, 6, 7]])
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        actual = encoder(ids, mask).last_hidden_state
        torch.testing.assert_close(actual[0, :2], encoder.tok_embeddings(ids[0, :2]))
        torch.testing.assert_close(actual[1], encoder.tok_embeddings(ids[1]))
        self.assertTrue(np.all(feeds[0]["inputs_embeds"][:, 2:] == 0))
        self.assertTrue(np.all(feeds[1]["inputs_embeds"][:, 3:] == 0))
        self.assertTrue(encoder.stats["zero_pad_embeddings"])

    def test_sparse_padding_is_rejected_before_inference(self):
        encoder = QNNEncoder(TinyEncoder(), {8: "model.onnx"})
        encoder._get_session = Mock()
        with self.assertRaisesRegex(ValueError, "contiguous right padding"):
            encoder(torch.tensor([[1, 0, 2]]), torch.tensor([[1, 0, 1]]))
        encoder._get_session.assert_not_called()


if __name__ == "__main__":
    unittest.main()
