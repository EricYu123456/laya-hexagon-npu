"""Small CPU tests; no model downloads or hardware sessions are required."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from npu.conv_linear import ConvLinear, qnn_conv_weight_overrides, replace_linears_with_conv
from npu.grouped_geglu import GroupedGeGLU
from npu.test_grouped_geglu import ReferenceGeGLU


HAS_ONNX = all(importlib.util.find_spec(name) is not None for name in ("onnx", "onnxruntime"))


class ConvLinearTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_float_projection_preserves_weights_bias_dtype_and_outputs(self):
        for bias in (False, True):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(bias=bias, dtype=dtype):
                    torch.manual_seed(201)
                    source = nn.Linear(41, 29, bias=bias, dtype=dtype).eval()
                    source.requires_grad_(False)
                    converted = ConvLinear(source)
                    self.assertFalse(converted.training)
                    self.assertEqual(converted.conv.weight.dtype, dtype)
                    self.assertFalse(converted.conv.weight.requires_grad)
                    torch.testing.assert_close(converted.weight, source.weight, rtol=0, atol=0)
                    self.assertNotEqual(converted.weight.data_ptr(), source.weight.data_ptr())
                    if bias:
                        torch.testing.assert_close(converted.bias, source.bias, rtol=0, atol=0)
                    for shape in ((1, 1, 41), (2, 127, 41), (1, 768, 41)):
                        inputs = torch.randn(shape, dtype=dtype)
                        torch.testing.assert_close(converted(inputs), source(inputs), rtol=0, atol=1e-5)
                        self.assertTrue(converted(inputs).is_contiguous())

    def test_recursive_grouped_geglu_conversion_and_shared_references(self):
        torch.manual_seed(202)
        original = ReferenceGeGLU(hidden=32, intermediate=49, bias=True).eval()
        maxima = torch.ones(49)
        maxima[[4, 30]] = torch.tensor([100.0, 16000.0])
        grouped = GroupedGeGLU.from_mlp(original, maxima)
        before = copy.deepcopy(grouped)
        count = replace_linears_with_conv(grouped)
        self.assertEqual(count, 9)  # gate, value, output for each of three groups
        self.assertFalse(any(isinstance(layer, nn.Linear) for layer in grouped.modules()))
        inputs = torch.randn(2, 73, 32)
        torch.testing.assert_close(grouped(inputs), original(inputs), rtol=0, atol=1e-4)
        torch.testing.assert_close(grouped(inputs), before(inputs), rtol=0, atol=1e-4)
        self.assertEqual(replace_linears_with_conv(grouped), 0)

        shared = nn.Linear(3, 4)
        container = nn.ModuleDict({"left": shared, "right": shared, "norm": nn.LayerNorm(4)})
        self.assertEqual(replace_linears_with_conv(container), 1)
        self.assertIs(container["left"], container["right"])
        self.assertIsInstance(container["norm"], nn.LayerNorm)

    def test_singleton_input_or_output_channels_and_noncontiguous_input(self):
        for incoming, outgoing in ((1, 17), (17, 1), (1, 1)):
            source = nn.Linear(incoming, outgoing, bias=True).eval()
            converted = ConvLinear(source)
            inputs = torch.randn(2, incoming, 11).transpose(1, 2)
            torch.testing.assert_close(converted(inputs), source(inputs), rtol=0, atol=1e-5)
        with self.assertRaises(ValueError):
            converted(torch.randn(2, 1))
        with self.assertRaises(ValueError):
            replace_linears_with_conv(nn.Linear(2, 3))

    @unittest.skipUnless(HAS_ONNX, "ONNX and ONNX Runtime are required for the tiny QDQ test")
    def test_onnx_qnn_config_uses_u16_activations_signed_channel_weights(self):
        import numpy as np
        import onnx
        import onnxruntime as ort
        from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize
        from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config

        class Reader(CalibrationDataReader):
            def __init__(self, inputs):
                self.rows = iter([{"input": inputs}])
            def get_next(self):
                return next(self.rows, None)

        # Unequal output-channel magnitudes make per-channel scales observable.
        torch.manual_seed(203)
        source = nn.Linear(7, 3, bias=True).eval()
        with torch.no_grad():
            source.weight.mul_(torch.tensor([0.03, 0.5, 3.0])[:, None])
        model = ConvLinear(source)
        inputs = torch.randn(1, 5, 7)
        with tempfile.TemporaryDirectory(prefix="laya-conv-test-") as directory:
            fp32_path = Path(directory) / "float.onnx"
            qdq_path = Path(directory) / "qdq.onnx"
            torch.onnx.export(model, inputs, str(fp32_path), input_names=["input"],
                              output_names=["output"], opset_version=20, dynamo=False)
            graph = onnx.load(fp32_path)
            self.assertEqual(sum(node.op_type == "Conv" for node in graph.graph.node), 1)
            self.assertFalse(any(node.op_type in {"MatMul", "Gemm"} for node in graph.graph.node))
            overrides = qnn_conv_weight_overrides(graph)
            config = get_qnn_qdq_config(
                graph, Reader(inputs.numpy()), activation_type=QuantType.QUInt16,
                weight_type=QuantType.QUInt8, per_channel=True,
                init_overrides=overrides, add_qtype_converts=False,
            )
            quantize(fp32_path, qdq_path, config)
            quantized = onnx.load(qdq_path)
            initializers = {item.name: item for item in quantized.graph.initializer}
            producers = {output: node for node in quantized.graph.node for output in node.output}
            conv = next(node for node in quantized.graph.node if node.op_type == "Conv")
            act_dq = producers[conv.input[0]]
            weight_dq = producers[conv.input[1]]
            self.assertEqual(initializers[act_dq.input[2]].data_type, onnx.TensorProto.UINT16)
            self.assertEqual(initializers[weight_dq.input[0]].data_type, onnx.TensorProto.INT8)
            axis = next(attr.i for attr in weight_dq.attribute if attr.name == "axis")
            self.assertEqual(axis, 0)
            scales = onnx.numpy_helper.to_array(initializers[weight_dq.input[1]])
            zero_points = onnx.numpy_helper.to_array(initializers[weight_dq.input[2]])
            self.assertEqual(scales.shape, (3,))
            self.assertGreater(scales.max() / scales.min(), 20)
            self.assertTrue(np.all(zero_points == 0))
            bias_dq = producers[conv.input[2]]
            self.assertEqual(initializers[bias_dq.input[0]].data_type, onnx.TensorProto.INT32)
            self.assertEqual(tuple(initializers[bias_dq.input[1]].dims), (3,))

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            floating = ort.InferenceSession(str(fp32_path), options, providers=["CPUExecutionProvider"])
            qdq = ort.InferenceSession(str(qdq_path), options, providers=["CPUExecutionProvider"])
            reference = source(inputs).detach().numpy()
            np.testing.assert_allclose(floating.run(None, {"input": inputs.numpy()})[0], reference, rtol=0, atol=1e-5)
            actual = qdq.run(None, {"input": inputs.numpy()})[0]
            self.assertLess(float(np.max(np.abs(actual - reference))), 0.05)


if __name__ == "__main__":
    unittest.main()
