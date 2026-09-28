"""Service startup tests with real FastAPI schemas and no model/NPU runtime."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


def import_service():
    torch = ModuleType('torch')
    torch.set_num_threads = Mock()
    torch.set_num_interop_threads = Mock()
    torch.get_num_threads = Mock(return_value=4)
    laya = ModuleType('laya')
    laya.load = Mock()
    npu = ModuleType('laya_npu')
    npu.load = Mock()
    path = Path(__file__).resolve().parents[1] / 'app.py'
    spec = importlib.util.spec_from_file_location('laya_service_readiness_test', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'torch': torch, 'laya': laya, 'laya_npu': npu}):
        spec.loader.exec_module(module)
    return module


class FakeAgent:
    def __init__(self, buckets=(768, 1024), *, npu=True, failure=None, fallback=False):
        self.cfg = {'max_len': 1024, 'head_max_len': 256}
        self.buckets = list(buckets)
        self.npu_stats = {'npu_calls': 0, 'cpu_fallbacks': 0}
        self.calls = []
        self.npu, self.failure, self.fallback = npu, failure, fallback
        self.during_predict = None

    def fidelity_metadata(self):
        return {'supported_buckets': self.buckets}

    def predict(self, state, questions):
        self.calls.append(copy.deepcopy((state, questions)))
        if self.during_predict is not None:
            self.during_predict()
        if self.failure:
            raise self.failure
        self.npu_stats['npu_calls'] += int(self.npu)
        self.npu_stats['cpu_fallbacks'] += int(self.fallback)
        return {'answers': {key: {'type': question['type']} for key, question in questions.items()},
                'usage': {'input_tokens': 10}}


class ServiceReadinessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.service = import_service()
        self.service.DEVICE = 'npu'
        self.service.CHECKPOINT = 'multilingual'
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.service.ROOT = Path(self.directory.name)
        (self.service.ROOT / 'models').mkdir()
        (self.service.ROOT / 'models/revision.txt').write_text('checkpoint-revision\n')

    def assert_not_ready(self):
        self.assertIsNone(self.service.agent)
        self.assertIsNone(self.service.readiness)
        with self.assertRaises(self.service.HTTPException) as error:
            self.service.health()
        self.assertEqual(error.exception.status_code, 503)

    async def test_npu_published_only_after_actual_warmup_and_cleared_on_shutdown(self):
        candidate = FakeAgent()
        candidate.during_predict = self.assert_not_ready
        self.service.laya_npu.load.return_value = candidate
        async with self.service.lifespan(self.service.app):
            self.assertIs(self.service.agent, candidate)
            self.assertEqual(len(candidate.calls), 1)
            self.assertEqual(candidate.calls[0], (self.service.WARMUP_STATE, self.service.WARMUP_QUESTIONS))
            self.assertEqual(self.service.health(), {
                'status': 'ready', 'checkpoint': 'multilingual', 'device': 'npu',
                'threads': 4, 'revision': 'checkpoint-revision', 'supported_input_capacity': 1024,
                'encoder_provider': 'QNNExecutionProvider', 'cpu_fallbacks': 0,
            })
            self.service.laya.load.assert_not_called()
        self.assert_not_ready()

    async def test_short_bucket_set_rejected_before_warmup(self):
        candidate = FakeAgent(buckets=(768,))
        self.service.laya_npu.load.return_value = candidate
        with self.assertRaisesRegex(RuntimeError, 'original Laya input capacity is 1024'):
            async with self.service.lifespan(self.service.app):
                self.fail('Incomplete capacity cannot enter service lifespan')
        self.assertEqual(candidate.calls, [])
        self.assert_not_ready()

    async def test_missing_npu_module_never_calls_cpu_loader(self):
        self.service.laya_npu = None
        self.service._npu_import_error = ImportError('provider unavailable')
        with self.assertRaisesRegex(RuntimeError, 'requires the strict laya_npu runtime'):
            async with self.service.lifespan(self.service.app):
                self.fail('Missing NPU runtime cannot enter service lifespan')
        self.service.laya.load.assert_not_called()
        self.assert_not_ready()

    async def test_warmup_backend_failure_leaves_no_ready_agent(self):
        self.service.laya_npu.load.return_value = FakeAgent(failure=RuntimeError('QNN context failed'))
        with self.assertRaisesRegex(RuntimeError, 'QNN context failed'):
            async with self.service.lifespan(self.service.app):
                self.fail('Failed warmup cannot enter service lifespan')
        self.assert_not_ready()

    async def test_warmup_requires_npu_call_and_zero_cpu_fallbacks(self):
        for candidate in (FakeAgent(npu=False), FakeAgent(fallback=True)):
            with self.subTest(npu=candidate.npu, fallback=candidate.fallback):
                self.service.laya_npu.load.return_value = candidate
                with self.assertRaisesRegex(RuntimeError, 'strict HTP inference'):
                    async with self.service.lifespan(self.service.app):
                        self.fail('A fallback or missing NPU call cannot qualify readiness')
                self.assert_not_ready()

    async def test_cpu_configuration_still_loads_upstream_and_warms(self):
        self.service.DEVICE = 'cpu'
        self.service.laya_npu = None
        candidate = FakeAgent(npu=False)
        candidate.during_predict = self.assert_not_ready
        self.service.laya.load.return_value = candidate
        async with self.service.lifespan(self.service.app):
            self.service.laya.load.assert_called_once_with(str(self.service.ROOT / 'models/multilingual'), device='cpu')
            self.assertEqual(len(candidate.calls), 1)
            self.assertEqual(self.service.health()['encoder_provider'], 'PyTorch')
            self.assertEqual(self.service.health()['supported_input_capacity'], 1024)
        self.assert_not_ready()

    async def test_predict_keeps_existing_response_contract(self):
        candidate = FakeAgent()
        self.service.laya_npu.load.return_value = candidate
        body = self.service.PredictRequest(state={'test': 'state'}, questions={
            'answer': {'type': 'choice', 'instructions': 'Choose.', 'criteria': ['yes', 'no']}})
        with self.assertRaises(self.service.HTTPException) as error:
            await self.service.predict(body)
        self.assertEqual(error.exception.status_code, 503)
        async with self.service.lifespan(self.service.app):
            result = await self.service.predict(body)
            self.assertEqual(result['answers'], {'answer': {'type': 'choice'}})
            self.assertEqual(result['usage'], {'input_tokens': 10})
            self.assertEqual(result['checkpoint'], 'multilingual')
            self.assertIsInstance(result['elapsed_ms'], float)
            self.assertEqual(candidate.calls[1][1], {'answer': {'type': 'choice', 'instructions': 'Choose.', 'criteria': ['yes', 'no']}})


if __name__ == '__main__':
    unittest.main()
