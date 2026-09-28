"""CPU-only contracts for backend-specific Conv calibration artifacts."""

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh

from npu import calibrate_conv_offsets as cal


def biased_conv_model():
    scales = np.array([.02, .03, .04], np.float32)
    arrays = {
        'as': np.array(.01, np.float32), 'az': np.array(32768, np.uint16),
        'wq': np.array([[1, -1], [2, 3], [-2, 1]], np.int8).reshape(3, 2, 1, 1),
        'ws': scales, 'wz': np.zeros(3, np.int8),
        'bq': np.array([100, 200, 300], np.int32),
        'bs': np.float32(.01) * scales, 'bz': np.zeros(3, np.int32),
        'os': np.array(.001, np.float32), 'oz': np.array(32768, np.uint16),
    }
    nodes = [
        h.make_node('QuantizeLinear', ['x', 'as', 'az'], ['xq'], domain='com.microsoft'),
        h.make_node('DequantizeLinear', ['xq', 'as', 'az'], ['xdq'], domain='com.microsoft'),
        h.make_node('DequantizeLinear', ['wq', 'ws', 'wz'], ['w'], domain='com.microsoft', axis=0),
        h.make_node('DequantizeLinear', ['bq', 'bs', 'bz'], ['b'], domain='com.microsoft', axis=0),
        h.make_node('Conv', ['xdq', 'w', 'b'], ['raw'], name='conv', kernel_shape=[1, 1]),
        h.make_node('QuantizeLinear', ['raw', 'os', 'oz'], ['yq'], domain='com.microsoft'),
        h.make_node('DequantizeLinear', ['yq', 'os', 'oz'], ['y'], domain='com.microsoft'),
    ]
    graph = h.make_graph(nodes, 'biased_conv', [h.make_tensor_value_info('x', T.FLOAT, [1, 2, 5, 1])],
                        [h.make_tensor_value_info('y', T.FLOAT, [1, 3, 5, 1])],
                        [nh.from_array(array, name) for name, array in arrays.items()])
    return h.make_model(graph, opset_imports=[h.make_opsetid('', 21), h.make_opsetid('com.microsoft', 1)], ir_version=10)


class ConvOffsetCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / 'source.onnx'
        self.target = self.root / 'corrected.onnx'
        onnx.save(biased_conv_model(), self.source)
        self.source_bytes = self.source.read_bytes()
        self.args = argparse.Namespace(source=str(self.source), out=str(self.root),
                                       validation_count=1, target=str(self.target))
        with contextlib.redirect_stdout(io.StringIO()):
            cal.prepare(self.args)
        self.manifest = json.loads((self.root / 'probe_manifest.json').read_text())
        self.result = {
            'source_sha256': cal.sha256(self.source), 'probe_sha256': self.manifest['probe_sha256'],
            'probe_manifest_sha256': cal.sha256(self.root / 'probe_manifest.json'),
            'all_shape_validations_passed': True,
            'runtime': {'test': 'synthetic CPU contract; no hardware measurement'},
            'shape_validation': [{'id': 'conv_000', 'passed': True, 'max_abs_offset_difference': 0., 'tolerance': .00025}],
        }
        self.correction = np.array([-.005, .008, -.003], np.float32)
        self.save_correction(self.correction)

    def save_result(self):
        cal.write_json(self.root / 'calibration_results.json', self.result)

    def save_correction(self, correction):
        np.savez(self.root / 'correction_biases.npz', conv_000=correction)
        self.result['correction_biases_sha256'] = cal.sha256(self.root / 'correction_biases.npz')
        self.save_result()

    def apply(self):
        with contextlib.redirect_stdout(io.StringIO()):
            cal.apply(self.args)

    def source_manifest(self, buckets=None):
        path = self.root / 'source_manifest.json'
        metadata = {
            'buckets': buckets or {'768': self.source.name},
            'model_sha256': {'768': cal.sha256(self.source)},
            'checkpoint_sha256': 'a' * 64, 'mask_penalty': -100.,
            'zero_pad_embeddings': True, 'matmul_rhs_refinement': 'all',
            'precision': 'a16w8', 'local_radius': 64,
        }
        cal.write_json(path, metadata)
        self.args.source_manifest = str(path)
        return path, metadata

    def assert_unchanged(self):
        self.assertEqual(self.source.read_bytes(), self.source_bytes)

    def test_existing_bias_is_preserved_and_correction_matches_cpu_arithmetic(self):
        self.apply()
        self.assert_unchanged()
        session = cal.runtime_session(self.target, False)
        output = session.run(None, {'x': np.zeros((1, 2, 5, 1), np.float32)})[0][0, :, 0, 0]
        scale = np.array([.02, .03, .04], np.float32) * np.float32(.01)
        old_bias = np.array([100, 200, 300]) * scale
        corrected = np.rint((old_bias + self.correction) / scale) * scale
        expected = np.rint(corrected / .001) * .001
        np.testing.assert_allclose(output, expected, atol=1e-6)
        source, target = onnx.load(self.source), onnx.load(self.target)
        self.assertEqual([v.SerializeToString() for v in source.graph.input], [v.SerializeToString() for v in target.graph.input])
        self.assertEqual([v.SerializeToString() for v in source.graph.output], [v.SerializeToString() for v in target.graph.output])
        old_initializers = {v.name: v.SerializeToString() for v in source.graph.initializer}
        for item in target.graph.initializer:
            if item.name in old_initializers:
                self.assertEqual(item.SerializeToString(), old_initializers[item.name])
        provenance = json.loads(self.target.with_suffix('.offsets.json').read_text())
        self.assertEqual(provenance['source_sha256'], cal.sha256(self.source))
        self.assertEqual(provenance['output_sha256'], cal.sha256(self.target))

    def test_same_path_and_existing_hardlink_target_cannot_mutate_source(self):
        self.args.target = str(self.source)
        with self.assertRaisesRegex(ValueError, 'separate output'):
            self.apply()
        self.args.target = str(self.target)
        os.link(self.source, self.target)
        with self.assertRaises(FileExistsError):
            self.apply()
        self.assert_unchanged()

    def test_source_hash_mismatch_is_rejected_before_output_is_written(self):
        changed = onnx.load(self.source)
        changed.doc_string = 'different source model'
        onnx.save(changed, self.source)
        with self.assertRaisesRegex(ValueError, 'exact source model'):
            self.apply()
        self.assertFalse(self.target.exists())

    def test_manifest_and_correction_hashes_are_enforced(self):
        with (self.root / 'probe_manifest.json').open('a') as stream:
            stream.write(' ')
        with self.assertRaisesRegex(ValueError, 'manifest checksum'):
            self.apply()
        cal.write_json(self.root / 'probe_manifest.json', self.manifest)
        self.result['probe_manifest_sha256'] = cal.sha256(self.root / 'probe_manifest.json')
        self.save_result()
        with (self.root / 'correction_biases.npz').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'array checksum'):
            self.apply()
        self.assertFalse(self.target.exists())
        self.assert_unchanged()

    def test_int32_overflow_is_rejected_instead_of_clipped(self):
        for value in (1e30, -1e30):
            with self.subTest(value=value):
                self.save_correction(np.full(3, value, np.float32))
                with self.assertRaisesRegex(ValueError, 'INT32 correction overflow'):
                    self.apply()
                self.assertFalse(self.target.exists())
                self.assert_unchanged()

    def test_nonfinite_or_wrong_length_corrections_are_rejected(self):
        for vector in (np.array([np.nan, 0, 0]), np.array([np.inf, 0, 0]), np.array([0, 0])):
            with self.subTest(vector=vector):
                self.save_correction(vector)
                with self.assertRaisesRegex(ValueError, 'Invalid correction vector'):
                    self.apply()
                self.assertFalse(self.target.exists())

    def test_missing_or_failed_spatial_controls_cannot_be_overridden_by_global_flag(self):
        for validation in ([], [{'id': 'conv_000', 'passed': False, 'max_abs_offset_difference': 1., 'tolerance': .001}]):
            with self.subTest(validation=validation):
                self.result['shape_validation'] = validation
                self.save_result()
                with self.assertRaisesRegex(ValueError, 'Spatial1'):
                    self.apply()
                self.assertFalse(self.target.exists())

    def test_v1_calibration_remains_usable_without_invented_new_fingerprints(self):
        self.manifest['format_version'] = 1
        cal.write_json(self.root / 'probe_manifest.json', self.manifest)
        self.result.pop('probe_manifest_sha256')
        self.save_result()
        self.apply()
        self.assert_unchanged()
        provenance = json.loads(self.target.with_suffix('.offsets.json').read_text())
        self.assertEqual(provenance['runtime'], self.result['runtime'])

    def test_record_tampering_is_detected_even_in_legacy_artifacts(self):
        self.manifest['format_version'] = 1
        self.manifest['records'][0]['activation_scale'] *= 2
        cal.write_json(self.root / 'probe_manifest.json', self.manifest)
        with self.assertRaisesRegex(ValueError, 'records do not match'):
            self.apply()
        self.assertFalse(self.target.exists())

    def test_prepare_refuses_to_replace_existing_calibration_artifacts(self):
        saved = (self.root / 'probe_manifest.json').read_bytes()
        with self.assertRaises(FileExistsError):
            cal.prepare(self.args)
        self.assertEqual((self.root / 'probe_manifest.json').read_bytes(), saved)
        self.assert_unchanged()

    def test_source_manifest_emits_only_corrected_bucket_and_preserves_policies(self):
        other = self.root / 'other.onnx'
        other.write_bytes(self.source_bytes)
        path, original = self.source_manifest({'768': self.source.name, '1024': other.name})
        original['model_sha256']['1024'] = cal.sha256(other)
        cal.write_json(path, original)
        before = path.read_bytes()
        self.apply()
        deployed = json.loads((self.root / 'manifest.json').read_text())
        self.assertEqual(deployed['buckets'], {'768': self.target.name})
        self.assertEqual(deployed['model_sha256'], {'768': cal.sha256(self.target)})
        correction = deployed['htp_conv_offset_correction']
        self.assertEqual(set(correction), {'768'})
        self.assertEqual(correction['768']['output_sha256'], cal.sha256(self.target))
        self.assertEqual(correction['768']['source_manifest_sha256'], cal.sha256(path))
        for key in ('checkpoint_sha256', 'mask_penalty', 'zero_pad_embeddings',
                    'matmul_rhs_refinement', 'precision', 'local_radius'):
            self.assertEqual(deployed[key], original[key])
        self.assertEqual(path.read_bytes(), before)
        self.assert_unchanged()

    def test_source_manifest_ambiguity_or_missing_match_is_rejected(self):
        for buckets in ({'768': self.source.name, '1024': self.source.name}, {'768': 'other.onnx'}):
            with self.subTest(buckets=buckets):
                (self.root / 'other.onnx').write_bytes(self.source_bytes)
                self.source_manifest(buckets)
                with self.assertRaisesRegex(ValueError, 'exactly one resolved path'):
                    self.apply()
                self.assertFalse(self.target.exists())
                self.assertFalse((self.root / 'manifest.json').exists())

    def test_source_manifest_checksum_or_already_corrected_marker_is_rejected(self):
        path, metadata = self.source_manifest()
        metadata['model_sha256']['768'] = '0' * 64
        cal.write_json(path, metadata)
        with self.assertRaisesRegex(ValueError, 'Source manifest SHA256'):
            self.apply()
        metadata['htp_conv_offset_correction'] = {}
        cal.write_json(path, metadata)
        with self.assertRaisesRegex(ValueError, 'already declares corrected'):
            self.apply()
        self.assertFalse(self.target.exists())

    def test_existing_destination_manifest_is_preserved(self):
        self.source_manifest()
        destination = self.root / 'manifest.json'
        destination.write_text('existing manifest')
        with self.assertRaisesRegex(FileExistsError, 'Destination manifest'):
            self.apply()
        self.assertEqual(destination.read_text(), 'existing manifest')
        self.assertFalse(self.target.exists())


if __name__ == '__main__':
    unittest.main()
