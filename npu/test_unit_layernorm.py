"""CPU equality and static-parameter contract checks for the V68 LN repair."""
import copy
import json
import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T, helper as h, numpy_helper as nh

from npu.unit_layernorm import repair_unit_layernorm_gamma


def model_fixture(gamma_type=np.uint16, gamma_value=65535, bias_value=0):
    dtype = np.dtype(gamma_type)
    maximum = np.iinfo(dtype).max
    tensors = {
        'xs': np.array(.01, np.float32), 'xz': np.array(32000, np.uint16),
        'gq': np.full(8, gamma_value, dtype), 'gs': np.array(1 / maximum, np.float32),
        'gz': np.array(0, dtype), 'bq': np.full(8, bias_value, np.int32),
        'bs': np.array([.01 / maximum], np.float32), 'bz': np.array(0, np.int32),
    }
    nodes = [
        h.make_node('QuantizeLinear', ['x', 'xs', 'xz'], ['xq'], name='xQ'),
        h.make_node('DequantizeLinear', ['xq', 'xs', 'xz'], ['xdq'], name='xDQ'),
        h.make_node('DequantizeLinear', ['gq', 'gs', 'gz'], ['gamma'], name='static_gamma'),
        h.make_node('Identity', ['gamma'], ['shared_gamma'], name='gamma_alias'),
        h.make_node('QuantizeLinear', ['shared_gamma', 'gs', 'gz'], ['alias_q'], name='aliasQ'),
        h.make_node('DequantizeLinear', ['alias_q', 'gs', 'gz'], ['alias_dq'], name='aliasDQ'),
        h.make_node('DequantizeLinear', ['bq', 'bs', 'bz'], ['beta'], name='zero_beta'),
        h.make_node('LayerNormalization', ['xdq', 'gamma', 'beta'], ['y0'], name='norm0', epsilon=1e-5),
        h.make_node('LayerNormalization', ['xdq', 'alias_dq', 'beta'], ['y1'], name='norm1', epsilon=1e-5),
    ]
    return h.make_model(h.make_graph(nodes, 'unit_ln', [h.make_tensor_value_info('x', T.FLOAT, [1, 3, 8])],
                                    [h.make_tensor_value_info(n, T.FLOAT, [1, 3, 8]) for n in ('y0', 'y1')],
                                    [nh.from_array(v, k) for k, v in tensors.items()]),
                        opset_imports=[h.make_opsetid('', 21)], ir_version=10)


def session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])


class UnitLayerNormTests(unittest.TestCase):
    def test_static_and_aliased_gamma_repair_is_cpu_bit_exact(self):
        model = model_fixture()
        before = session(model)
        original_initializers = {i.name: i.SerializeToString() for i in model.graph.initializer}
        original_io = ([x.SerializeToString() for x in model.graph.input], [x.SerializeToString() for x in model.graph.output])
        metadata = repair_unit_layernorm_gamma(model)
        self.assertEqual(metadata['repaired_layer_norms'], 2)
        self.assertGreater(metadata['removed_dead_parameter_nodes'], 0)
        json.dumps(metadata, allow_nan=False)
        onnx.checker.check_model(model)
        after = session(model)
        rng = np.random.default_rng(7)
        for x in (np.zeros((1, 3, 8), np.float32), np.ones((1, 3, 8), np.float32),
                  rng.normal(size=(1, 3, 8)).astype(np.float32),
                  np.full((1, 3, 8), 500, np.float32)):
            for actual, expected in zip(after.run(None, {'x': x}), before.run(None, {'x': x})):
                np.testing.assert_array_equal(actual, expected)
        values = {i.name: nh.to_array(i) for i in model.graph.initializer}
        producers = {o: n for n in model.graph.node for o in n.output}
        for node in model.graph.node:
            if node.op_type == 'LayerNormalization':
                g, b = producers[node.input[1]], producers[node.input[2]]
                self.assertEqual(values[g.input[0]].dtype, np.uint8)
                np.testing.assert_array_equal(values[g.input[0]].astype(np.float32) * values[g.input[1]], np.ones(8))
                self.assertEqual(values[b.input[0]].dtype, np.int32)
                self.assertEqual(float(values[b.input[1]][0]), float(np.float32(.01) * np.float32(1 / 255)))
        for item in model.graph.initializer:
            if item.name in original_initializers:
                self.assertEqual(item.SerializeToString(), original_initializers[item.name])
        self.assertEqual(original_io, ([x.SerializeToString() for x in model.graph.input], [x.SerializeToString() for x in model.graph.output]))
        repaired_bytes = model.SerializeToString()
        self.assertEqual(repair_unit_layernorm_gamma(model)['repaired_layer_norms'], 0)
        self.assertEqual(model.SerializeToString(), repaired_bytes)

    def test_existing_u8_gamma_is_not_changed(self):
        model = model_fixture(np.uint8, 255)
        before = model.SerializeToString()
        self.assertEqual(repair_unit_layernorm_gamma(model)['repaired_layer_norms'], 0)
        self.assertEqual(model.SerializeToString(), before)

    def test_nonunit_or_nonzero_folded_parameters_fail_without_partial_mutation(self):
        for model in (model_fixture(gamma_value=65534), model_fixture(bias_value=1)):
            before = model.SerializeToString()
            with self.assertRaises(ValueError):
                repair_unit_layernorm_gamma(model)
            self.assertEqual(model.SerializeToString(), before)

    def test_live_aliased_parameter_outputs_are_preserved(self):
        model = model_fixture()
        model.graph.output.append(h.make_tensor_value_info('alias_dq', T.FLOAT, [8]))
        before = session(model)
        repair_unit_layernorm_gamma(model)
        after = session(model)
        x = np.arange(24, dtype=np.float32).reshape(1, 3, 8)
        for actual, expected in zip(after.run(None, {'x': x}), before.run(None, {'x': x})):
            np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
