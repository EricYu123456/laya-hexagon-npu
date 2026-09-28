"""Input-only checks for predeclared long probes; no model answers or weights."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import make_long_probe_suite as generator


class LongProbeSuiteTests(unittest.TestCase):
    def setUp(self):
        self.questions = {
            'first': {'type': 'choice', 'instructions': 'Decide.', 'criteria': {'a': 'A', 'b': 'B'}},
            'second': {'type': 'noul', 'instructions': 'Determine whether it applies.'},
        }
        self.rows = [{'state': 'FORBIDDEN UNSELECTED ROW', 'questions': ''} for _ in range(28)]
        self.rows[2] = {'state': json.dumps({'note': '保留列二'}), 'questions': json.dumps(self.questions), 'workflow': 'alpha'}
        self.rows[27] = {'state': json.dumps(['保留列二十七']), 'questions': json.dumps(self.questions), 'workflow': 'beta'}
        self.calls = []

        def build(tokenizer, state, question, max_len, head_max_len):
            self.calls.append((tokenizer, state, copy.deepcopy(question), max_len, head_max_len))
            rate = 128 if question['type'] == 'choice' else 64
            return list(range(min(max_len, rate * len(state.split('\n\n'))))), [0, 1]

        self.build_sequence = Mock(side_effect=build)
        laya = ModuleType('laya')
        common = ModuleType('laya.common')
        common.build_sequence = self.build_sequence
        laya.common = common
        patcher = patch.dict(sys.modules, {'laya': laya, 'laya.common': common})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tokenizer = object()
        self.convert = Mock(side_effect=lambda question: copy.deepcopy(question))

    def make_suite(self, indices=(2, 27)):
        return generator.make_suite(self.rows, list(indices), self.tokenizer, self.convert, 1024, 256)

    def test_only_declared_rows_all_questions_and_original_budgets(self):
        originals = copy.deepcopy(self.rows)
        suite = self.make_suite()
        self.assertEqual([probe['source_index'] for probe in suite], [2, 27])
        self.assertEqual([probe['id'] for probe in suite], ['heldout-long-2', 'heldout-long-27'])
        self.assertEqual([probe['workflow'] for probe in suite], ['alpha', 'beta'])
        for probe in suite:
            self.assertEqual(probe['questions'], self.questions)
            self.assertEqual(probe['min_sequence_tokens'], 1024)
            self.assertNotIn('answers', probe)
            self.assertNotIn('output', probe)
            # Slower-growing second question requires16 repetitions; filling only
            # the first question (8 repetitions) must not finish generation.
            self.assertEqual(len(probe['state'].split('\n\n')), 16)
        for tokenizer, state, question, max_len, head_max_len in self.calls:
            self.assertIs(tokenizer, self.tokenizer)
            self.assertEqual((max_len, head_max_len), (1024, 256))
            self.assertIn(question, self.questions.values())
            self.assertNotIn('FORBIDDEN', state)
            self.assertIn('保留列', state)
            self.assertNotIn('\\u4fdd', state)
        self.assertEqual(self.rows, originals)

    def test_generation_is_deterministic(self):
        self.assertEqual(self.make_suite(), self.make_suite())

    def test_invalid_or_duplicate_selection_rejected(self):
        for indices in ([], [2, 2], [-1], [28], [True], [2.0]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                self.make_suite(indices)
        self.build_sequence.assert_not_called()

    def test_empty_or_unsupported_state_rejected_before_tokenization(self):
        self.build_sequence.side_effect = None
        self.build_sequence.return_value = ([0] * 1024, [0])
        for state in ('', '  \t', {}, [], None, 1, True):
            with self.subTest(state=state):
                self.rows[2]['state'] = json.dumps(state)
                with self.assertRaises(ValueError):
                    self.make_suite([2])
        self.build_sequence.assert_not_called()

    def test_every_question_must_equal_budget_even_if_builder_misbehaves(self):
        self.build_sequence.side_effect = lambda tok, state, q, *limits: ([0] * (1024 if q['type'] == 'choice' else 1025), [0])
        with self.assertRaisesRegex(ValueError, 'did not fill'):
            self.make_suite([2])
        self.assertEqual(self.build_sequence.call_count, 24)

    def test_empty_questions_or_malformed_json_rejected(self):
        self.rows[2]['questions'] = '{}'
        with self.assertRaises(ValueError):
            self.make_suite([2])
        self.rows[2]['state'] = 'malformed JSON'
        with self.assertRaises(json.JSONDecodeError):
            self.make_suite([2])
        self.build_sequence.assert_not_called()

    def test_existing_metadata_evidence_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'suite.json'
            metadata = output.with_suffix('.metadata.json')
            metadata.write_text('original evidence')
            args = ['make_long_probe_suite.py', '--dataset-path', 'unused', '--output', str(output)]
            with patch.object(sys, 'argv', args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                generator.main()
            self.assertEqual(metadata.read_text(), 'original evidence')
            self.assertFalse(output.exists())

    def test_declared_plan_is_disjoint_and_discloses_previous_source_exposure(self):
        path = Path(__file__).resolve().parents[1] / 'reports/evaluation-plans/long-input-heldout.json'
        plan = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(plan['kind'], 'long_input_validation_plan')
        self.assertEqual(plan['format_version'], 1)
        self.assertEqual(plan['indices'], list(range(2, 400, 25)))
        self.assertEqual(plan['excluded_calibration_and_development_indices'],
                         sorted([*range(0, 400, 25), *range(1, 400, 25)]))
        self.assertFalse(set(plan['indices']) & set(plan['excluded_calibration_and_development_indices']))
        self.assertEqual(plan['original_input_limits'], {'max_len': 1024, 'head_max_len': 256})
        self.assertIn('untransformed', plan['purpose'])
        self.assertIn('768', plan['purpose'])


if __name__ == '__main__':
    unittest.main()
