"""Calibrate QNN HTP pointwise Conv offsets using only all-zero inputs.

prepare: build independent Conv probes; no runtime or accelerator access.
run: strict HTP/CPU zero probes and original-spatial-shape validation.
apply: add backend-specific INT32 biases to a separate model artifact.

Never changes the source model. Calibration does not use evaluation examples.
The correction is deliberately HTP-specific and changes CPU graph semantics.
See npu/CONV_OFFSET_CALIBRATION.md for the workflow and validation requirements.
"""
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def scalar(array):
    if np.asarray(array).size != 1:
        raise ValueError('Expected scalar activation or output encoding')
    return np.asarray(array).reshape(-1)[0].item()


def index(model):
    return ({out: n for n in model.graph.node for out in n.output},
            {i.name: i for i in model.graph.initializer})


def conv_records(model):
    prods, initializers = index(model)
    arrays = {name: nh.to_array(value) for name, value in initializers.items()}
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in [*model.graph.input, *model.graph.value_info, *model.graph.output]}

    def shape(name):
        if name in shapes and all(shapes[name]):
            return shapes[name]
        node = prods.get(name)
        if node is not None and node.op_type in {'QuantizeLinear', 'DequantizeLinear', 'Identity', 'Cast'}:
            return shape(node.input[0])
        raise ValueError(f'Cannot infer original static Conv input shape: {name}')

    records, names = [], set()
    for node in model.graph.node:
        if node.op_type != 'Conv':
            continue
        if not node.name or node.name in names:
            raise ValueError('Conv nodes must have unique, nonempty names')
        names.add(node.name)
        attrs = {a.name: h.get_attribute_value(a) for a in node.attribute}
        if (attrs.get('kernel_shape', [1, 1]) != [1, 1] or attrs.get('group', 1) != 1
                or attrs.get('strides', [1, 1]) != [1, 1]
                or attrs.get('dilations', [1, 1]) != [1, 1]
                or attrs.get('pads', [0, 0, 0, 0]) != [0, 0, 0, 0]
                or attrs.get('auto_pad', b'NOTSET') not in (b'NOTSET', b'VALID')):
            raise ValueError(f'Unsupported non-pointwise or grouped Conv: {node.name}')
        adq, wdq = prods[node.input[0]], prods[node.input[1]]
        if adq.op_type != 'DequantizeLinear' or wdq.op_type != 'DequantizeLinear':
            raise ValueError(f'{node.name}: expected quantized activation and weights')
        if arrays[adq.input[2]].dtype != np.uint16 or arrays[wdq.input[2]].dtype != np.int8:
            raise ValueError(f'{node.name}: expected A16U / W8S')
        weights = arrays[wdq.input[0]]
        if list(weights.shape[2:]) != [1, 1]:
            raise ValueError(f'{node.name}: non-pointwise weights')
        output_q = [n for n in model.graph.node if n.op_type == 'QuantizeLinear' and n.input[0] == node.output[0]]
        if len(output_q) != 1:
            raise ValueError(f'{node.name}: expected one output quantizer')
        oq = output_q[0]
        if arrays[oq.input[2]].dtype != np.uint16:
            raise ValueError(f'{node.name}: expected U16 output')
        weight_scale = arrays[wdq.input[1]]
        if weight_scale.size not in (1, weights.shape[0]):
            raise ValueError(f'{node.name}: incompatible per-channel weight encoding')
        if weight_scale.size > 1 and next((a.i for a in wdq.attribute if a.name == 'axis'), 1) != 0:
            raise ValueError(f'{node.name}: weight quantization must use output-channel axis 0')
        for encoding in [arrays[adq.input[1]], weight_scale, arrays[oq.input[1]]]:
            if not np.isfinite(encoding).all() or not (encoding > 0).all():
                raise ValueError(f'{node.name}: quantization scales must be finite and positive')
        dims = shape(node.input[0])
        if len(dims) != 4 or dims[0] != 1 or dims[1] != weights.shape[1]:
            raise ValueError(f'{node.name}: expected static NCHW batch 1 input matching weights')
        records.append({
            'id': f'conv_{len(records):03d}', 'node': node.name,
            'input_shape': dims, 'output_channels': int(weights.shape[0]),
            'activation_scale': float(scalar(arrays[adq.input[1]])),
            'activation_zero': int(scalar(arrays[adq.input[2]])),
            'output_scale': float(scalar(arrays[oq.input[1]])),
            'output_zero': int(scalar(arrays[oq.input[2]])),
            'has_bias': len(node.input) == 3,
        })
    if not records:
        raise ValueError('No eligible Conv nodes')
    return records


def prepare(args):
    source, out = Path(args.source).resolve(), Path(args.out).resolve()
    if args.validation_count < 1:
        raise ValueError('validation_count must be positive')
    if source == out / 'zero_probes.onnx':
        raise ValueError('Probe output must not overwrite its source model')
    if any((out / name).exists() for name in ('zero_probes.onnx', 'probe_manifest.json', 'calibration_results.json')):
        raise FileExistsError(f'Calibration artifacts already exist; choose a new output directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    model = onnx.load(source)
    records = conv_records(model)
    prods, initials = index(model)
    named = {n.name: n for n in model.graph.node}
    inputs, outputs, nodes, kept_initials = [], [], [], {}
    copied_static = set()

    def keep_static(name):
        if name in initials:
            kept_initials[name] = initials[name]
        elif name not in copied_static:
            original = prods[name]
            if original.op_type not in {'DequantizeLinear', 'Identity', 'Cast', 'Constant'}:
                raise ValueError(f'Unsupported dynamic bias/weight dependency: {name}')
            for dependency in original.input:
                keep_static(dependency)
            nodes.append(copy.deepcopy(original))
            copied_static.update(original.output)

    def add_probe(record, original_shape=False):
        node = named[record['node']]
        adq = prods[node.input[0]]
        oq = next(n for n in model.graph.node if n.op_type == 'QuantizeLinear' and n.input[0] == node.output[0])
        prefix = record['id'] + ('_original' if original_shape else '_small')
        dims = list(record['input_shape']) if original_shape else [1, record['input_shape'][1], 1, 1]
        odims = [dims[0], record['output_channels'], dims[2], dims[3]]
        for key in [*adq.input[1:], *oq.input[1:]]:
            keep_static(key)
        for key in node.input[1:]:
            keep_static(key)
        inputs.append(h.make_tensor_value_info(prefix + '_x', T.FLOAT, dims))
        nodes.extend([
            h.make_node('QuantizeLinear', [prefix + '_x', *adq.input[1:]], [prefix + '_xq'], name=prefix + '_Q', domain=adq.domain),
            h.make_node('DequantizeLinear', [prefix + '_xq', *adq.input[1:]], [prefix + '_xdq'], name=prefix + '_DQ', domain=adq.domain),
        ])
        conv = copy.deepcopy(node)
        conv.name = prefix + '_Conv'
        conv.input[0] = prefix + '_xdq'
        conv.output[0] = prefix + '_raw'
        nodes.append(conv)
        nodes.extend([
            h.make_node('QuantizeLinear', [prefix + '_raw', *oq.input[1:]], [prefix + '_yq'], name=prefix + '_outQ', domain=oq.domain),
            h.make_node('DequantizeLinear', [prefix + '_yq', *oq.input[1:]], [prefix + '_y'], name=prefix + '_outDQ', domain=oq.domain),
        ])
        outputs.append(h.make_tensor_value_info(prefix + '_y', T.FLOAT, odims))

    for record in records:
        add_probe(record)
    validation = [r for r in records if np.prod(r['input_shape'][2:]) > 1][:args.validation_count]
    if not validation:
        raise ValueError('Need at least one original spatial shape >1 validation')
    for record in validation:
        add_probe(record, original_shape=True)
    probe = h.make_model(h.make_graph(nodes, 'independent_zero_conv_probes', inputs, outputs, list(kept_initials.values())),
                         opset_imports=list(model.opset_import), ir_version=model.ir_version)
    onnx.checker.check_model(probe)
    path = out / 'zero_probes.onnx'
    onnx.save(probe, path)
    metadata = {'format_version': 2, 'source_path': str(source), 'source_sha256': sha256(source),
                'probe_sha256': sha256(path), 'calibration_inputs': 'all-zero; no evaluation samples',
                'records': records, 'shape_validation_ids': [r['id'] for r in validation],
                'warning': 'Offsets measured after output QDQ have up to one output-step uncertainty; spatial validation is sampled.'}
    write_json(out / 'probe_manifest.json', metadata)
    print(json.dumps({'prepared_convs': len(records), 'original_shape_validations': len(validation), 'path': str(path)}), flush=True)


def runtime_session(path, npu):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    if not npu:
        return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    import onnxruntime_qnn as qnn
    options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
    options.add_provider_for_devices([d for d in ort.get_ep_devices() if d.ep_name == 'QNNExecutionProvider'],
                                    {'backend_path': qnn.get_qnn_htp_path(), 'htp_arch': '68', 'htp_performance_mode': 'burst'})
    return ort.InferenceSession(str(path), options)


def run(args):
    import onnxruntime as ort
    import onnxruntime_qnn as qnn
    if __package__:
        from .fidelity_runtime import qnn_backend_fingerprint
    else:
        from fidelity_runtime import qnn_backend_fingerprint
    out = Path(args.out).resolve()
    manifest = json.loads((out / 'probe_manifest.json').read_text(encoding='utf-8'))
    path = out / 'zero_probes.onnx'
    if sha256(path) != manifest['probe_sha256']:
        raise ValueError('Probe model checksum changed')
    ort.register_execution_provider_library('QNNExecutionProvider', qnn.get_library_path())
    backend = Path(qnn.get_qnn_htp_path()).resolve()
    runtime = qnn_backend_fingerprint(ort.__version__, importlib.metadata.version('onnxruntime-qnn'),
                                      backend, htp_arch=68, require_auxiliary=True)
    runtime.update({'platform': platform.platform(), 'cpu_ep_fallback': False})
    arrays = {}
    for use_npu, key in [(False, 'cpu'), (True, 'htp')]:
        print('RUN', key, flush=True)
        sess = runtime_session(path, use_npu)
        feeds = {i.name: np.zeros(i.shape, np.float32) for i in sess.get_inputs()}
        values = sess.run(None, feeds)
        if any(not np.isfinite(value).all() for value in values):
            raise ValueError(f'{key}: non-finite Conv probe output')
        arrays.update({key + '_' + o.name: value for o, value in zip(sess.get_outputs(), values)})
        del sess, values, feeds
        gc.collect()
    offsets, summary, validation = {}, [], []
    for record in manifest['records']:
        key = record['id']
        c, n = arrays['cpu_' + key + '_small_y'], arrays['htp_' + key + '_small_y']
        delta = (n - c).reshape(-1).astype(np.float32)
        offsets[key] = -delta
        summary.append({'id': key, 'node': record['node'], 'mean_abs_offset': float(np.abs(delta).mean()),
                        'max_abs_offset': float(np.abs(delta).max()), 'output_step': record['output_scale']})
        if key in manifest['shape_validation_ids']:
            co, no = arrays['cpu_' + key + '_original_y'], arrays['htp_' + key + '_original_y']
            difference = (no - co) - delta.reshape(1, -1, 1, 1)
            maximum = float(np.abs(difference).max())
            tolerance = record['output_scale'] * .25 + 1e-7
            validation.append({'id': key, 'max_abs_offset_difference': maximum, 'tolerance': tolerance,
                               'passed': bool(maximum <= tolerance)})
    result = {'source_sha256': manifest['source_sha256'], 'probe_sha256': manifest['probe_sha256'],
              'probe_manifest_sha256': sha256(out / 'probe_manifest.json'),
              'runtime': runtime, 'shape_validation': validation, 'all_shape_validations_passed': all(v['passed'] for v in validation),
              'per_conv': summary, 'warning': manifest['warning']}
    np.savez(out / 'zero_outputs.npz', **arrays)
    np.savez(out / 'correction_biases.npz', **offsets)
    result['correction_biases_sha256'] = sha256(out / 'correction_biases.npz')
    write_json(out / 'calibration_results.json', result)
    print(json.dumps({'convs': len(summary), 'shape_validation': validation}), flush=True)


def static_value(name, producers, initials):
    if name in initials:
        return nh.to_array(initials[name]).astype(np.float64)
    node = producers[name]
    if node.op_type != 'DequantizeLinear':
        raise ValueError(f'Unsupported existing bias producer: {node.op_type}')
    q, scale, zero = (nh.to_array(initials[k]) for k in node.input)
    if scale.size > 1:
        axis = next((a.i for a in node.attribute if a.name == 'axis'), 1)
        shape = [1] * q.ndim
        shape[axis] = scale.size
        scale, zero = scale.reshape(shape), zero.reshape(shape)
    return (q.astype(np.float64) - zero) * scale


def deployment_manifest(source_manifest, source, target, source_digest):
    """Preflight an optional single-bucket deployment manifest without writes."""
    if not source_manifest:
        return None
    if __package__:
        from .fidelity_runtime import load_bucket_manifest
    else:
        from fidelity_runtime import load_bucket_manifest
    source_manifest = Path(source_manifest).resolve()
    metadata = json.loads(source_manifest.read_text(encoding='utf-8'))
    if 'htp_conv_offset_correction' in metadata:
        raise ValueError('Source manifest already declares corrected models; use an uncorrected source manifest')
    destination = target.parent / 'manifest.json'
    if destination.exists():
        raise FileExistsError(f'Destination manifest already exists: {destination}')
    paths, metadata = load_bucket_manifest(source_manifest)
    matches = [bucket for bucket, path in paths.items() if path == source]
    if len(matches) != 1:
        raise ValueError(f'Source manifest must have exactly one resolved path matching --source; found {len(matches)}')
    bucket = str(matches[0])
    hashes = metadata.get('model_sha256')
    expected = hashes.get(bucket) if isinstance(hashes, dict) else None
    if not isinstance(expected, str) or expected.lower() != source_digest:
        raise ValueError(f'Source manifest SHA256 for bucket {bucket} does not match --source')
    return {'bucket': bucket, 'metadata': copy.deepcopy(metadata), 'destination': destination,
            'source_manifest_sha256': sha256(source_manifest)}


def apply(args):
    source, out, target = Path(args.source).resolve(), Path(args.out).resolve(), Path(args.target).resolve()
    if target.suffix.lower() != '.onnx':
        raise ValueError('Corrected target must use an .onnx extension')
    if source == target:
        raise ValueError('A separate output model path is required')
    if target.exists() or target.with_suffix('.offsets.json').exists():
        raise FileExistsError(f'Corrected output already exists; choose a new target: {target}')
    manifest = json.loads((out / 'probe_manifest.json').read_text(encoding='utf-8'))
    result = json.loads((out / 'calibration_results.json').read_text(encoding='utf-8'))
    source_digest = sha256(source)
    if source_digest != manifest['source_sha256'] or result['source_sha256'] != manifest['source_sha256']:
        raise ValueError('Calibration does not match exact source model')
    deployment = deployment_manifest(getattr(args, 'source_manifest', None), source, target, source_digest)
    if result.get('probe_sha256') != manifest['probe_sha256']:
        raise ValueError('Calibration does not match the prepared probe model')
    if manifest.get('format_version', 1) >= 2:
        if result.get('probe_manifest_sha256') != sha256(out / 'probe_manifest.json'):
            raise ValueError('Probe manifest checksum changed after calibration')
    validation = result.get('shape_validation', [])
    expected_ids = manifest['shape_validation_ids']
    if (not expected_ids or len(validation) != len(expected_ids)
            or {v['id'] for v in validation} != set(expected_ids)
            or not result.get('all_shape_validations_passed')
            or any(not v.get('passed') or not np.isfinite(v['max_abs_offset_difference'])
                   or v['max_abs_offset_difference'] > v['tolerance'] for v in validation)):
        raise ValueError('Spatial1 and original-shape offsets differ; calibration cannot be applied')
    if sha256(out / 'correction_biases.npz') != result['correction_biases_sha256']:
        raise ValueError('Correction array checksum changed')
    with np.load(out / 'correction_biases.npz', allow_pickle=False) as saved:
        biases = {key: saved[key].copy() for key in saved.files}
    model = onnx.load(source)
    if conv_records(model) != manifest['records']:
        raise ValueError('Conv calibration records do not match the source model')
    if set(biases) != {record['id'] for record in manifest['records']}:
        raise ValueError('Correction arrays do not cover exactly the calibrated Conv nodes')
    prods, initials = index(model)
    records = {r['node']: r for r in manifest['records']}
    occupied = {i.name for i in model.graph.initializer} | set(prods)
    nodes, rounding = [], []
    for node in model.graph.node:
        if node.name not in records:
            nodes.append(node)
            continue
        record = records[node.name]
        prefix = '__htp_offset_' + record['id']
        if any(name.startswith(prefix) for name in occupied):
            raise ValueError('Offset correction already applied or naming collision')
        wdq = prods[node.input[1]]
        ws = nh.to_array(initials[wdq.input[1]]).astype(np.float32)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            bs = np.broadcast_to(ws * np.float32(record['activation_scale']), (record['output_channels'],)).copy()
        if not np.isfinite(bs).all() or not (bs > 0).all():
            raise ValueError(f'Invalid INT32 bias scale: {node.name}')
        old = static_value(node.input[2], prods, initials).reshape(-1) if len(node.input) == 3 else np.zeros(record['output_channels'])
        correction = biases[record['id']]
        if correction.shape != (record['output_channels'],) or not np.isfinite(correction).all():
            raise ValueError(f'Invalid correction vector: {node.name}')
        requested = old + correction
        with np.errstate(over='ignore', invalid='ignore'):
            quantized = np.rint(requested / bs)
        limits = np.iinfo(np.int32)
        if (not np.isfinite(quantized).all() or (quantized < limits.min).any()
                or (quantized > limits.max).any()):
            raise ValueError(f'INT32 correction overflow: {node.name}')
        bq = quantized.astype(np.int32)
        model.graph.initializer.extend([nh.from_array(bq, prefix + '_q'), nh.from_array(bs, prefix + '_scale'),
                                        nh.from_array(np.zeros_like(bq), prefix + '_zero')])
        nodes.append(h.make_node('DequantizeLinear', [prefix + '_q', prefix + '_scale', prefix + '_zero'], [prefix + '_dq'],
                                 name=prefix + '_DQ', axis=0, domain=wdq.domain))
        changed = copy.deepcopy(node)
        if len(changed.input) == 3:
            changed.input[2] = prefix + '_dq'
        else:
            changed.input.append(prefix + '_dq')
        nodes.append(changed)
        rounding.append({'id': record['id'], 'max_abs_bias_rounding': float(np.max(np.abs(requested - bq.astype(np.float64) * bs)))})
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    onnx.checker.check_model(model)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, target)
    provenance = {'source_sha256': manifest['source_sha256'], 'output_sha256': sha256(target),
                  'runtime': result['runtime'], 'calibration_results_sha256': sha256(out / 'calibration_results.json'),
                  'calibration_inputs': manifest['calibration_inputs'], 'shape_validation': result['shape_validation'],
                  'corrected_convs': len(rounding), 'bias_rounding': rounding,
                  'warning': 'Backend-specific HTP correction changes CPU semantics. Revalidate full-model fidelity on HTP; per-Conv output QDQ uncertainty is up to one output step.'}
    if 'calibration_reuse' in result:
        provenance['calibration_reuse'] = result['calibration_reuse']
    if deployment:
        provenance['source_manifest_sha256'] = deployment['source_manifest_sha256']
    write_json(target.with_suffix('.offsets.json'), provenance)
    if deployment:
        bucket, metadata = deployment['bucket'], deployment['metadata']
        metadata['buckets'] = {bucket: target.name}
        metadata['model_sha256'] = {bucket: provenance['output_sha256']}
        metadata['htp_conv_offset_correction'] = {bucket: provenance}
        write_json(deployment['destination'], metadata)
    print(json.dumps({'corrected_convs': len(rounding), 'output': str(target)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='Build zero probes from an immutable QDQ model; no NPU used')
    p.add_argument('--source', required=True, help='Uncorrected static A16U/W8S pointwise-Conv ONNX model')
    p.add_argument('--out', required=True, help='Directory for the probe model and calibration records')
    p.add_argument('--validation-count', type=int, default=3,
                   help='Positive count of non-unit spatial Conv shapes to compare with spatial1 (default: 3)')
    p = sub.add_parser('run', help='Run CPU and strict QNN HTP zero probes on the target Pi')
    p.add_argument('--out', required=True, help='Prepared probe directory; source npu/env.sh before running')
    p = sub.add_parser('apply', help='Write a separate HTP-corrected ONNX model after validation')
    p.add_argument('--source', required=True, help='Exact uncorrected ONNX file used by prepare')
    p.add_argument('--out', required=True, help='Completed probe directory including calibration_results.json')
    p.add_argument('--target', required=True, help='New corrected ONNX path; existing targets are rejected')
    p.add_argument('--source-manifest', help='Optional uncorrected deployment manifest; emit a new single-bucket manifest.json beside --target')
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'apply': apply}[args.command](args)


if __name__ == '__main__':
    main()
