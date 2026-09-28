"""HTTP verifier control-flow tests; no HTTP server or accelerator is contacted."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmark_fidelity import json_hash, sha256_file
import verify_service as verifier


class ServiceVerifierTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.question = {'type': 'choice', 'instructions': 'Choose.', 'criteria': ['a', 'b']}
        self.answer = {'model': 'laya-rl-agent', 'answers': {'q': {'type': 'choice', 'choice': 'a',
                       'probabilities': {'a': .9, 'b': .1}, 'confidence': .8, 'action': {'act_probability': 1}}}}
        self.suite = [{'id': label, 'state': label, 'questions': {'q': self.question}}
                      for label in ['short', 'long', 'short-again']]
        self.records = [{'id': p['id'], 'input_sha256': json_hash([p['state'], p['questions']]),
                         'tokens': {'q': {'tokens': 1024 if p['id'] == 'long' else 32}},
                         'output': {**copy.deepcopy(self.answer), 'usage': {'input_tokens': 1024 if p['id'] == 'long' else 32, 'output_tokens': 0}},
                         'npu_execution_verified': True} for p in self.suite]
        self.reference = {'kind': 'laya_probe_reference', 'backend': 'cpu', 'module': 'laya',
                          'suite_sha256': json_hash(self.suite), 'model_files': {'weights': 'same'}, 'records': self.records}
        self.npu_records = copy.deepcopy(self.records)
        for record in self.npu_records:
            bucket = '1024' if record['id'] == 'long' else '768'
            record.update(tokens_equal=True, reported_tokens_equal=True,
                          accelerator_delta={'npu_calls': 1, 'cpu_fallbacks': 0, 'bucket_calls': {bucket: 1}},
                          expected_bucket_calls={bucket: 1})
        self.expected = {**self.reference, 'backend': 'npu', 'kind': 'laya_probe_comparison', 'passed': True,
                         'records': self.npu_records, 'candidate_assets': {'supported_buckets': [768, 1024]},
                         'accelerator_stats_after_measurement': {'npu_calls': 3, 'cpu_fallbacks': 0,
                                                                  'bucket_calls': {'768': 2, '1024': 1}}}
        self.health = {'status': 'ready', 'device': 'npu', 'checkpoint': 'multilingual',
                       'encoder_provider': 'QNNExecutionProvider', 'supported_input_capacity': 1024,
                       'supported_buckets': [768, 1024], 'npu_calls': 1, 'bucket_calls': {'768': 1},
                       'cpu_fallbacks': 0, 'context_cache_hits': 1, 'context_cache_misses': 0, 'context_cache_errors': 0}
        self.wrong_bucket = False
        self.http_calls = []

    def fake_http(self, base, endpoint, body=None, timeout=120):
        self.http_calls.append((endpoint, body))
        if endpoint == '/health':
            return 200, copy.deepcopy(self.health), .1
        index = next((i for i, p in enumerate(self.suite) if p['state'] == body['state']), None)
        if index is None:
            return 422, {'detail': 'validation error'}, .2
        bucket = '1024' if body['state'] == 'long' and not self.wrong_bucket else '768'
        self.health['npu_calls'] += 1
        self.health['bucket_calls'][bucket] = self.health['bucket_calls'].get(bucket, 0) + 1
        return 200, {**copy.deepcopy(self.npu_records[index]['output']), 'elapsed_ms': 1.0, 'checkpoint': 'multilingual'}, 1.5

    def run_verifier(self, expected=True):
        for name, data in [('suite', self.suite), ('reference', self.reference)]:
            (self.root / f'{name}.json').write_text(json.dumps(data), encoding='utf-8')
        self.expected.setdefault('reference_sha256', sha256_file(self.root/'reference.json'))
        (self.root/'expected.json').write_text(json.dumps(self.expected), encoding='utf-8')
        args = ['--suite', str(self.root/'suite.json'), '--reference', str(self.root/'reference.json'),
                '--output', str(self.root/'output.json'), '--ready-timeout', '.001']
        if expected:
            args += ['--expected-npu', str(self.root/'expected.json')]
        with patch.object(verifier, 'http_json', side_effect=self.fake_http), contextlib.redirect_stdout(io.StringIO()):
            code = verifier.main(args)
        return code, json.loads((self.root/'output.json').read_text(encoding='utf-8'))

    def test_valid_service_proves_counter_transitions_and_exact_npu_replay(self):
        code, report = self.run_verifier()
        self.assertEqual(code, 0)
        self.assertTrue(report['passed'])
        self.assertTrue(report['integration_passed'])
        self.assertTrue(report['development_fidelity_passed'])
        self.assertTrue(report['long_short_transition_verified'])
        self.assertTrue(all(r['standalone_npu_outputs_equal'] for r in report['records']))
        self.assertEqual(report['health_after']['npu_calls'], 4)
        self.assertEqual(len(report['invalid_requests']), 5)
        self.assertTrue(all(r['http_status'] == 422 and r['accelerator_delta']['npu_calls'] == 0 for r in report['invalid_requests']))

    def test_wrong_bucket_is_not_hidden_by_correct_answers(self):
        self.wrong_bucket = True
        code, report = self.run_verifier()
        self.assertEqual(code, 1)
        self.assertFalse(report['passed'])
        self.assertFalse(report['records'][1]['npu_execution_verified'])

    def test_standalone_answer_change_fails_even_when_decision_is_same(self):
        self.expected = copy.deepcopy(self.expected)
        self.expected['records'][0]['output']['answers']['q']['confidence'] = .81
        code, report = self.run_verifier()
        self.assertEqual(code, 1)
        self.assertFalse(report['records'][0]['standalone_npu_outputs_equal'])

    def test_missing_precompiled_context_is_not_ready_for_qualification(self):
        self.health['context_cache_misses'] = 1
        with patch.object(verifier.time, 'sleep'):
            code, report = self.run_verifier()
        self.assertEqual(code, 1)
        self.assertFalse(report['records'])
        self.assertIn('readiness deadline', report['error'])

    def test_five_question_cli_suite_rejected_before_http(self):
        self.suite[0]['questions'] = {str(i): self.question for i in range(5)}
        self.reference['suite_sha256'] = json_hash(self.suite)
        with self.assertRaisesRegex(ValueError, 'HTTP schema'):
            self.run_verifier(expected=False)
        self.assertEqual(self.http_calls, [])

    def test_failed_development_fidelity_is_retained_while_exact_replay_passes_integration(self):
        self.expected['passed'] = False
        self.npu_records[1]['output']['answers']['q'].update(choice='b', probabilities={'a': .1, 'b': .9})
        code, report = self.run_verifier()
        self.assertEqual(code, 0)
        self.assertTrue(report['passed'])
        self.assertTrue(report['integration_passed'])
        self.assertFalse(report['development_fidelity_passed'])
        self.assertFalse(report['standalone_development_fidelity_passed'])
        self.assertAlmostEqual(report['overall']['decision_error_percent'], 100/3)
        self.assertTrue(all(r['standalone_npu_outputs_equal'] for r in report['records']))

    def test_missing_standalone_report_cannot_certify_exact_replay(self):
        code, report = self.run_verifier(expected=False)
        self.assertEqual(code, 1)
        self.assertTrue(report['runtime_checks_passed'])
        self.assertTrue(report['development_fidelity_passed'])
        self.assertFalse(report['integration_passed'])
        self.assertIn('--expected-npu', report['integration_not_verified_reason'])

    def test_foreign_reference_report_rejected_before_http(self):
        self.expected['reference_sha256'] = 'a'*64
        with self.assertRaisesRegex(ValueError, 'exact reference'):
            self.run_verifier()
        self.assertEqual(self.http_calls, [])

    def test_token_mismatch_not_hidden_by_true_execution_flag(self):
        self.npu_records[1]['tokens']['q']['tokens'] = 1000
        with self.assertRaisesRegex(ValueError, 'input/token identity'):
            self.run_verifier()
        self.assertEqual(self.http_calls, [])

    def test_stale_execution_flag_does_not_replace_per_record_counter_proof(self):
        self.npu_records[1]['accelerator_delta']['npu_calls'] = 0
        with self.assertRaisesRegex(ValueError, 'actual encoder execution'):
            self.run_verifier()
        self.assertEqual(self.http_calls, [])

    def test_aggregate_counter_mismatch_rejected_before_http(self):
        self.expected['accelerator_stats_after_measurement']['cpu_fallbacks'] = 1
        with self.assertRaisesRegex(ValueError, 'aggregate execution counters'):
            self.run_verifier()
        self.assertEqual(self.http_calls, [])


if __name__ == '__main__':
    unittest.main()
