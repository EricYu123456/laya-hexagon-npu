"""Reserved-data/length invariants, with no model imports or inference."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def import_builder_without_model_libraries():
    names = ['numpy', 'onnx', 'onnxruntime', 'torch', 'torch.nn', 'laya', 'laya.common',
             'onnxruntime.quantization', 'onnxruntime.quantization.execution_providers',
             'onnxruntime.quantization.execution_providers.qnn']
    modules = {name: ModuleType(name) for name in names}
    modules['torch'].nn = modules['torch.nn']
    modules['torch.nn'].Module = object
    modules['torch'].no_grad = lambda: lambda function: function
    quant = modules['onnxruntime.quantization']
    quant.CalibrationDataReader = object
    quant.QuantType = SimpleNamespace()
    quant.quantize = Mock(side_effect=AssertionError('No quantization in selection tests'))
    modules['onnxruntime.quantization.execution_providers.qnn'].get_qnn_qdq_config = Mock()
    modules['laya.common'].build_sequence = Mock()
    source = Path(__file__).resolve().parents[1] / 'npu/build_fidelity.py'
    spec = importlib.util.spec_from_file_location('long_calibration_builder_test', source)
    builder = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(builder)
    return builder


class TrackingRows(list):
    def __init__(self, rows):
        super().__init__(rows)
        self.accesses = []

    def __getitem__(self, index):
        self.accesses.append(index)
        return super().__getitem__(index)


class LongCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.builder = import_builder_without_model_libraries()
        # Unselected rows are deliberately malformed: using even one would fail.
        rows = [{'state': 'DO NOT READ EVALUATION DATA', 'questions': ''} for _ in range(76)]
        self.questions = {
            'first': {'type': 'choice', 'instructions': 'Decide.', 'criteria': {'a': 'A', 'b': 'B'}},
            'second': {'type': 'noul', 'instructions': 'Is the condition met?'},
        }
        rows[0] = {'state': json.dumps({'note': 'calibration-row0'}), 'questions': json.dumps(self.questions)}
        rows[50] = {'state': json.dumps(['calibration-row50']), 'questions': json.dumps(self.questions)}
        self.rows = TrackingRows(rows)
        self.pq = ModuleType('pyarrow.parquet')
        self.pq.read_table = Mock(return_value=SimpleNamespace(to_pylist=lambda: self.rows))
        pyarrow = ModuleType('pyarrow')
        pyarrow.parquet = self.pq
        patcher = patch.dict(sys.modules, {'pyarrow': pyarrow, 'pyarrow.parquet': self.pq})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.agent = SimpleNamespace(cfg={'max_len': 1024, 'head_max_len': 256}, tok=object(),
                                     _to_internal=Mock(side_effect=lambda q: {'original_question': copy.deepcopy(q)}))
        self.calls = []

        def build(tokenizer, state, question, max_len, head_max_len):
            self.calls.append((tokenizer, state, question, max_len, head_max_len))
            text = state if isinstance(state, str) else json.dumps(state)
            length = min(max_len, 128 * text.count('calibration-row'))
            return list(range(length)), [0, 4]

        self.builder.build_sequence = Mock(side_effect=build)
        self.selected = [(0, 'first', [10, 11]), (50, 'second', [20, 21]), (0, 'second', [30, 31])]

    def run_long(self, selected=None, target=1024):
        return self.builder.long_calibration_sequences(self.agent, 'pinned-calibration.parquet',
                                                       self.selected if selected is None else selected, target)

    def test_only_selected_rows_questions_and_original_budgets_are_used(self):
        original = copy.deepcopy(self.selected)
        result = self.run_long()
        self.assertEqual(self.rows.accesses, [0, 50, 0])
        self.assertEqual([(i, q) for i, q, _ in result],
                         [(0, 'first/repeated-state'), (50, 'second/repeated-state'), (0, 'second/repeated-state')])
        self.assertTrue(all(len(ids) == 1024 for _, _, ids in result))
        self.assertEqual(self.selected, original)
        self.assertEqual(self.agent.cfg, {'max_len': 1024, 'head_max_len': 256})
        allowed_texts = {json.dumps({'note': 'calibration-row0'}, ensure_ascii=False),
                         json.dumps(['calibration-row50'], ensure_ascii=False)}
        for tokenizer, state, question, max_len, head_max_len in self.calls:
            self.assertIs(tokenizer, self.agent.tok)
            self.assertEqual((max_len, head_max_len), (1024, 256))
            self.assertTrue(all(part in allowed_texts for part in state.split('\n\n')))
            self.assertIn(question['original_question'], self.questions.values())
        # The builder appends new variants; every original question/id sequence survives.
        combined = original + result
        self.assertEqual(combined[:len(original)], original)
        self.assertEqual(len(combined), 2 * len(original))

    def test_other_bucket_rejected_before_reading_dataset(self):
        with self.assertRaisesRegex(ValueError, 'original max_len'):
            self.run_long(target=768)
        self.pq.read_table.assert_not_called()

    def test_original_sequences_reject_ambiguous_indices_and_preserve_rows(self):
        for indices in ([], [0, 0], [-1], [76], [True], [0.0]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                self.builder.sequences(self.agent, 'pinned-calibration.parquet', indices)
        self.assertEqual(self.rows.accesses, [])
        originals = self.builder.sequences(self.agent, 'pinned-calibration.parquet', [0, 50])
        self.assertEqual(self.rows.accesses, [0, 50])
        self.assertEqual([(i, q) for i, q, _ in originals],
                         [(0, 'first'), (0, 'second'), (50, 'first'), (50, 'second')])
        self.assertEqual(self.calls[0][1], {'note': 'calibration-row0'})
        self.assertEqual(self.calls[2][1], ['calibration-row50'])

    def test_empty_string_states_rejected(self):
        for state in ('', '  \n\t'):
            with self.subTest(state=state):
                self.rows[0]['state'] = json.dumps(state)
                with self.assertRaisesRegex(ValueError, 'empty'):
                    self.run_long(selected=[self.selected[0]])

    def test_nonempty_string_state_is_repeated_without_json_wrapping(self):
        self.rows[0]['state'] = json.dumps('calibration-row0')
        result = self.run_long(selected=[self.selected[0]])
        self.assertEqual(len(result[0][2]), 1024)
        self.assertEqual(self.calls[0][1], 'calibration-row0')

    def test_empty_structured_or_invalid_state_rejected(self):
        for state in ({}, [], None, 1, True):
            with self.subTest(state=state):
                self.rows[0]['state'] = json.dumps(state)
                with self.assertRaises(ValueError):
                    self.run_long(selected=[self.selected[0]])

    def test_invalid_selected_indices_cannot_alias_evaluation_rows(self):
        for index in (-1, 76, True, 0.0):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.run_long(selected=[(index, 'first', [1])])

    def test_sequence_builder_that_never_fills_target_fails_boundedly(self):
        self.builder.build_sequence = Mock(return_value=([1] * 1023, [0]))
        with self.assertRaisesRegex(ValueError, 'did not fill 1024'):
            self.run_long(selected=[self.selected[0]])
        self.assertEqual(self.builder.build_sequence.call_count, 12)

    def test_malformed_json_or_missing_question_fails(self):
        self.rows[0]['state'] = '{invalid JSON'
        with self.assertRaises(json.JSONDecodeError):
            self.run_long(selected=[self.selected[0]])
        self.rows[0]['state'] = json.dumps('calibration-row0')
        with self.assertRaises(KeyError):
            self.run_long(selected=[(0, 'nonexistent', [1])])


if __name__ == '__main__':
    unittest.main()
