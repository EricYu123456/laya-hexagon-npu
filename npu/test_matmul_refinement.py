"""CPU numerical and graph-contract tests for experimental RHS refinement."""

import copy
import json
import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T, helper as h, numpy_helper as nh

from npu.matmul_refinement import (
    _covering_encoding, attention_matmul_refinement, refine_dynamic_matmul_rhs,
)


def tiny_model(rhs_scale=1 / 32768, high_scale=0.01, output_scale=1 / 32768, static_rhs=False):
    nodes, initializers = [], []

    def qdq(name, role, scale, zero, dtype):
        initializers.extend([nh.from_array(np.array(scale, np.float32), role + '_s'),
                             nh.from_array(np.array(zero, dtype), role + '_z')])
        nodes.extend([h.make_node('QuantizeLinear', [name, role + '_s', role + '_z'], [role + '_q'], name=role + '_Q', domain='com.microsoft'),
                      h.make_node('DequantizeLinear', [role + '_q', role + '_s', role + '_z'], [role + '_dq'], name=role + '_DQ', domain='com.microsoft')])
        return role + '_dq'

    a = qdq('a', 'a16', 1 / 65535, 0, np.uint16)
    b16 = qdq('b', 'b16', rhs_scale, 32768, np.uint16)
    b8 = qdq(b16, 'b8', high_scale, 128, np.uint8)
    nodes.append(h.make_node('MatMul', [a, b8], ['y'], name='attention_pv'))
    out = qdq('y', 'y16', output_scale, 32768, np.uint16)
    inputs = [h.make_tensor_value_info('a', T.FLOAT, [2, 64])]
    if static_rhs:
        initializers.append(nh.from_array(np.ones((64, 7), np.float32) * .1, 'b'))
    else:
        inputs.append(h.make_tensor_value_info('b', T.FLOAT, [64, 7]))
    model = h.make_model(h.make_graph(nodes, 'tiny_refinement', inputs,
                                    [h.make_tensor_value_info(out, T.FLOAT, [2, 7])], initializers),
                        opset_imports=[h.make_opsetid('', 21), h.make_opsetid('com.microsoft', 1)], ir_version=10)
    return onnx.shape_inference.infer_shapes(model)


def run(model, inputs):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider']).run(None, inputs)[0]


def attention_model():
    model = tiny_model()
    # Model the actual export path, where Cast hides Softmax from a direct
    # producer lookup. Use names without attention/PV hints to test topology.
    nodes = list(model.graph.node)
    nodes[0].input[0] = 'cast_probabilities'
    nodes.insert(0, h.make_node('Cast', ['identity_probabilities'], ['cast_probabilities'], name='cast', to=T.FLOAT))
    nodes.insert(0, h.make_node('Identity', ['probabilities'], ['identity_probabilities'], name='identity'))
    nodes.insert(0, h.make_node('Softmax', ['a'], ['probabilities'], name='normalize', axis=-1))
    for node in nodes:
        if node.name == 'attention_pv':
            node.name = 'projection_7'
    nodes.extend([
        h.make_node('QuantizeLinear', ['a', 'a16_s', 'a16_z'], ['qk_a_q'], name='qk_a_Q', domain='com.microsoft'),
        h.make_node('DequantizeLinear', ['qk_a_q', 'a16_s', 'a16_z'], ['qk_a_dq'], name='qk_a_DQ', domain='com.microsoft'),
        h.make_node('MatMul', ['qk_a_dq', 'b8_dq'], ['qk_y'], name='projection_8'),
        h.make_node('QuantizeLinear', ['qk_y', 'y16_s', 'y16_z'], ['qk_y_q'], name='qk_y_Q', domain='com.microsoft'),
        h.make_node('DequantizeLinear', ['qk_y_q', 'y16_s', 'y16_z'], ['qk_y_dq'], name='qk_y_DQ', domain='com.microsoft'),
    ])
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.output.append(h.make_tensor_value_info('qk_y_dq', T.FLOAT, [2, 7]))
    return model


class MatMulRefinementTests(unittest.TestCase):
    def test_reduces_error_near_coarse_quantization_boundaries(self):
        rng = np.random.default_rng(123)
        a = rng.uniform(.1, 1, (2, 64)).astype(np.float32)
        a /= a.sum(axis=-1, keepdims=True)
        for offset in (.49, .51):
            with self.subTest(offset=offset):
                b = ((rng.integers(-60, 60, (64, 7)) + offset) * .01).astype(np.float32)
                model = tiny_model()
                before = run(model, {'a': a, 'b': b})
                original_io = ([x.SerializeToString() for x in model.graph.input], [x.SerializeToString() for x in model.graph.output])
                original_initializers = {x.name: x.SerializeToString() for x in model.graph.initializer}
                metadata = refine_dynamic_matmul_rhs(model, lhs_l1_bounds={'attention_pv': 1.01})
                self.assertEqual(json.loads(json.dumps(metadata, allow_nan=False)), metadata)
                onnx.checker.check_model(model)
                after = run(model, {'a': a, 'b': b})
                expected = a @ b
                self.assertLess(np.linalg.norm(after - expected), np.linalg.norm(before - expected) * .1)
                self.assertLess(np.max(np.abs(after - expected)), 1e-4)
                self.assertEqual(metadata['refined_matmuls'], 1)
                self.assertEqual(original_io, ([x.SerializeToString() for x in model.graph.input], [x.SerializeToString() for x in model.graph.output]))
                for item in model.graph.initializer:
                    if item.name in original_initializers:
                        self.assertEqual(item.SerializeToString(), original_initializers[item.name])
                self.assertEqual(refine_dynamic_matmul_rhs(model)['refined_matmuls'], 0)

    def test_clipping_tail_bounds_and_expanded_high_range(self):
        rng = np.random.default_rng(42)
        a = rng.uniform(.1, 1, (2, 64)).astype(np.float32)
        a /= a.sum(axis=-1, keepdims=True)
        b = rng.uniform(-8, 8, (64, 7)).astype(np.float32)
        model = tiny_model(rhs_scale=10 / 32768, high_scale=.01, output_scale=1 / 1024)
        before = run(model, {'a': a, 'b': b})
        retained = copy.deepcopy(model)
        metadata = refine_dynamic_matmul_rhs(retained, lhs_l1_bounds={'attention_pv': 1.01})['nodes'][0]
        all_values = (np.arange(65536, dtype=np.float64) - 32768) * np.float32(10 / 32768)
        coarse = (np.clip(np.rint(all_values / np.float32(.01)) + 128, 0, 255) - 128) * np.float32(.01)
        residual = all_values - coarse
        self.assertLessEqual(metadata['residual_range'][0], residual.min())
        self.assertGreaterEqual(metadata['residual_range'][1], residual.max())
        expanded = refine_dynamic_matmul_rhs(model, lhs_l1_bounds={'attention_pv': 1.01}, expand_high_range=True)['nodes'][0]
        self.assertTrue(expanded['expanded_high_range'])
        self.assertLess(expanded['residual_scale'], metadata['residual_scale'] / 100)
        after = run(model, {'a': a, 'b': b})
        self.assertLess(np.linalg.norm(after - a @ b), np.linalg.norm(before - a @ b) * .01)

    def test_only_requested_dynamic_conversion_pattern_is_changed(self):
        static = tiny_model(static_rhs=True)
        before = static.SerializeToString()
        self.assertEqual(refine_dynamic_matmul_rhs(static)['refined_matmuls'], 0)
        self.assertEqual(static.SerializeToString(), before)
        dynamic = tiny_model()
        before = dynamic.SerializeToString()
        self.assertEqual(refine_dynamic_matmul_rhs(dynamic, node_names={'other'})['refined_matmuls'], 0)
        self.assertEqual(dynamic.SerializeToString(), before)

    def test_high_partial_saturation_keeps_compensating_residual(self):
        a = np.full((2, 64), 1 / 64, np.float32)
        # The high term rounds to +/-0.2, outside final +/-0.18. Its
        # residual must be added before final saturation, including when
        # that residual brings an otherwise saturated value back inside.
        for value in (.17, -.17, .19, -.19):
            with self.subTest(value=value):
                model = tiny_model(high_scale=.1, output_scale=.18 / 32768)
                b = np.full((64, 7), value, np.float32)
                metadata = refine_dynamic_matmul_rhs(
                    model, lhs_l1_bounds={'attention_pv': 1.01}
                )['nodes'][0]
                bound = metadata['partial_output_bound']
                self.assertLessEqual(metadata['high_output_range'][0], -.18 - bound)
                self.assertGreaterEqual(metadata['high_output_range'][1], .18 * 32767 / 32768 + bound)
                after = run(model, {'a': a, 'b': b})
                expected = np.clip(a @ b, -.18, .18 * 32767 / 32768)
                self.assertLess(np.max(np.abs(after - expected)), 3e-4)

    def test_all_sub_operands_and_partial_outputs_have_u16_encodings(self):
        model = tiny_model()
        refine_dynamic_matmul_rhs(model)
        producers = {output: node for node in model.graph.node for output in node.output}
        consumers = {name: node for node in model.graph.node for name in node.input}
        values = {item.name: nh.to_array(item) for item in model.graph.initializer}
        for node in model.graph.node:
            if node.op_type == 'Sub':
                for name in node.input:
                    dq = producers[name]
                    self.assertEqual(dq.op_type, 'DequantizeLinear')
                    self.assertEqual(values[dq.input[2]].dtype, np.uint16)
            if node.op_type == 'MatMul':
                self.assertEqual(values[producers[node.input[0]].input[2]].dtype, np.uint16)
                self.assertEqual(values[producers[node.input[1]].input[2]].dtype, np.uint8)
                self.assertEqual(values[consumers[node.output[0]].input[2]].dtype, np.uint16)

    def test_range_encoding_preserves_tiny_opposite_sign_tail(self):
        for low, high in [(-1e-9, 10), (-10, 1e-9), (-1, 1), (0, 2), (-2, 0)]:
            scale, zero = _covering_encoding(low, high, 255)
            self.assertLessEqual(-zero * float(scale), low)
            self.assertGreaterEqual((255 - zero) * float(scale), high)

    def test_attention_scope_uses_softmax_cast_identity_topology(self):
        for scope, expected_count in [('pv', 1), ('all', 2)]:
            with self.subTest(scope=scope):
                model = attention_model()
                metadata = attention_matmul_refinement(model, scope=scope)
                self.assertEqual(metadata['refined_matmuls'], expected_count)
                self.assertEqual(len(metadata['pv_bounds']), 1)
                self.assertEqual(metadata['pv_bounds'][0]['node'], 'projection_7')
                self.assertEqual(metadata['pv_bounds'][0]['sequence_length'], 64)
                self.assertAlmostEqual(metadata['pv_bounds'][0]['lhs_l1_bound'], 1 + 64 / (2 * 65535) + .01)
                if scope == 'all':
                    qk_record = next(item for item in metadata['nodes'] if item['node'] == 'projection_8')
                    self.assertAlmostEqual(qk_record['lhs_l1_bound'], 64, places=5)
                self.assertEqual(json.loads(json.dumps(metadata, allow_nan=False)), metadata)
                onnx.checker.check_model(model)
                result = run(model, {'a': np.zeros((2, 64), np.float32), 'b': np.full((64, 7), .1749, np.float32)})
                self.assertLess(np.max(np.abs(result - .1749)), 1e-4)

    def test_attention_tight_bound_rejects_unverified_assumptions(self):
        model = attention_model()
        before = model.SerializeToString()
        with self.assertRaisesRegex(ValueError, 'scope'):
            attention_matmul_refinement(model, scope='unknown')
        self.assertEqual(model.SerializeToString(), before)
        for kind in ('axis', 'scale', 'cast'):
            with self.subTest(kind=kind):
                model = attention_model()
                if kind == 'axis':
                    next(node for node in model.graph.node if node.name == 'normalize').attribute[0].i = 0
                elif kind == 'cast':
                    next(node for node in model.graph.node if node.name == 'cast').attribute[0].i = T.FLOAT16
                else:
                    next(item for item in model.graph.initializer if item.name == 'a16_s').CopyFrom(
                        nh.from_array(np.array(.001, np.float32), 'a16_s'))
                before = model.SerializeToString()
                with self.assertRaises(ValueError):
                    attention_matmul_refinement(model, scope='pv')
                self.assertEqual(model.SerializeToString(), before)

    def test_attention_accepts_subunity_nonnegative_probability_range(self):
        model = attention_model()
        scale = np.float32(.75 / 65535)
        next(item for item in model.graph.initializer if item.name == 'a16_s').CopyFrom(
            nh.from_array(np.array(scale, np.float32), 'a16_s'))
        metadata = attention_matmul_refinement(model, scope='pv')
        record = metadata['pv_bounds'][0]
        self.assertEqual(record['probability_scale'], float(scale))
        self.assertAlmostEqual(record['lhs_l1_bound'], 1 + 64 * float(scale) / 2 + .01)
        self.assertEqual(metadata['refined_matmuls'], 1)
        onnx.checker.check_model(model)
        # Saturation of the first row reduces its probability mass below one;
        # the second row exercises ordinary rounding of nonsaturated values.
        logits = np.zeros((2, 64), np.float32)
        logits[0, 0] = 80
        values = np.full((64, 7), .1749, np.float32)
        probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        rounded = np.clip(np.rint(probabilities / scale), 0, 65535) * scale
        self.assertTrue(np.all(rounded.sum(axis=-1) <= record['lhs_l1_bound']))
        result = run(model, {'a': logits, 'b': values})
        self.assertLess(np.max(np.abs(result - rounded @ values)), 1e-4)

    def test_attention_rejects_negative_nonfinite_or_zero_probability_encoding(self):
        for scale, zero in [(-1 / 65535, 0), (0, 0), (np.nan, 0), (np.inf, 0), (1 / 65535, 1)]:
            with self.subTest(scale=scale, zero=zero):
                model = attention_model()
                for name, value, dtype in [('a16_s', scale, np.float32), ('a16_z', zero, np.uint16)]:
                    next(item for item in model.graph.initializer if item.name == name).CopyFrom(
                        nh.from_array(np.array(value, dtype), name))
                before = model.SerializeToString()
                with self.assertRaises(ValueError):
                    attention_matmul_refinement(model, scope='pv')
                self.assertEqual(model.SerializeToString(), before)

    def test_attention_accumulates_distinct_qdq_stages_and_collapses_equal_ones(self):
        base_scale = np.float32(1 / 65535)
        for final_scale in (base_scale, np.float32(.9998 / 65535)):
            with self.subTest(final_scale=final_scale):
                model = attention_model()
                model.graph.initializer.extend([
                    nh.from_array(np.array(final_scale, np.float32), 'final_s'),
                    nh.from_array(np.array(0, np.uint16), 'final_z'),
                ])
                nodes = list(model.graph.node)
                index = next(i for i, item in enumerate(nodes) if item.name == 'projection_7')
                nodes[index].input[0] = 'final_dq'
                nodes[index:index] = [
                    h.make_node('Cast', ['a16_dq'], ['recast_p'], name='recast_p', to=T.FLOAT),
                    h.make_node('QuantizeLinear', ['recast_p', 'final_s', 'final_z'], ['final_q'], name='final_Q', domain='com.microsoft'),
                    h.make_node('DequantizeLinear', ['final_q', 'final_s', 'final_z'], ['final_dq'], name='final_DQ', domain='com.microsoft'),
                ]
                del model.graph.node[:]
                model.graph.node.extend(nodes)
                metadata = attention_matmul_refinement(model, scope='pv')
                record = metadata['pv_bounds'][0]
                scales = [float(base_scale)] if final_scale == base_scale else [float(base_scale), float(final_scale)]
                self.assertEqual(record['probability_rounding_scales'], scales)
                self.assertEqual(record['probability_quantize_stages'], 2)
                self.assertAlmostEqual(record['lhs_l1_bound'], 1 + 64 * sum(scales) / 2 + .01)
                onnx.checker.check_model(model)
                result = run(model, {'a': np.zeros((2, 64), np.float32), 'b': np.full((64, 7), .1749, np.float32)})
                self.assertLess(np.max(np.abs(result - .1749)), 1e-4)

    def test_attention_rejects_mismatched_q_and_dq_within_one_pair(self):
        model = attention_model()
        model.graph.initializer.append(nh.from_array(np.array(.9 / 65535, np.float32), 'bad_dq_scale'))
        next(node for node in model.graph.node if node.name == 'a16_DQ').input[1] = 'bad_dq_scale'
        before = model.SerializeToString()
        with self.assertRaisesRegex(ValueError, 'pair must have matching encodings'):
            attention_matmul_refinement(model, scope='pv')
        self.assertEqual(model.SerializeToString(), before)


if __name__ == '__main__':
    unittest.main()
