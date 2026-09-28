#!/usr/bin/env python3
"""Run AFTER exclusive NPU qualification/deployment; no model is loaded here.

Run on Pi beside the deployed repository, against an otherwise idle service.
Example:
  .venv/bin/python verify_service.py --reference .work/probe_reference_wsl.json \
      --expected-npu .work/final_probe_npu.json --output .work/service-verification.json
This is service integration evidence, not a replacement for held-out fidelity.
``passed`` means integration only; ``development_fidelity_passed`` separately
applies the unchanged <=5% decision-error and mean-TV <=5% development gates.
An expected standalone NPU report is required to certify exact API replay.
Capture systemd/cgroup memory and deployed artifact hashes separately.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from benchmark_fidelity import decision_metrics, json_hash, sha256_file, summarize_decisions


def http_json(base, endpoint, body=None, timeout=120):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode('utf-8')
    request = Request(base.rstrip('/') + endpoint, data=data,
                      headers={'Content-Type': 'application/json'} if data is not None else {})
    started = time.monotonic()
    try:
        with urlopen(request, timeout=timeout) as response:
            status, raw = response.status, response.read()
    except HTTPError as error:
        status, raw = error.code, error.read()
    return status, json.loads(raw), (time.monotonic() - started) * 1000


def health(base, timeout):
    status, body, _ = http_json(base, '/health', timeout=timeout)
    if status != 200 or body.get('status') != 'ready' or body.get('device') != 'npu':
        raise ValueError(f'Expected ready NPU health, got HTTP {status}: {body}')
    if body.get('encoder_provider') != 'QNNExecutionProvider' or body.get('supported_input_capacity', 0) < 1024:
        raise ValueError('Health does not confirm QNN and original1024 capacity')
    if body.get('cpu_fallbacks') != 0 or body.get('context_cache_errors') != 0 or body.get('context_cache_misses') != 0 or body.get('npu_calls', 0) < 1:
        raise ValueError('Health does not confirm strict warmup from prepared contexts without cache errors')
    if not body.get('supported_buckets') or max(body['supported_buckets']) < 1024:
        raise ValueError('Health lacks a complete bucket set')
    for key in ['npu_calls', 'bucket_calls', 'context_cache_hits', 'context_cache_misses', 'context_cache_errors']:
        if key not in body:
            raise ValueError(f'Missing health counter: {key}')
    return body


def delta(before, after):
    counts = {str(key): value - before['bucket_calls'].get(str(key), 0)
              for key, value in after['bucket_calls'].items()
              if value - before['bucket_calls'].get(str(key), 0)}
    return {'npu_calls': after['npu_calls'] - before['npu_calls'], 'bucket_calls': counts,
            **{key: after[key] - before[key] for key in
               ['cpu_fallbacks', 'context_cache_hits', 'context_cache_misses', 'context_cache_errors']}}


def load_inputs(args):
    suite = json.loads(args.suite.read_text(encoding='utf-8'))
    ref = json.loads(args.reference.read_text(encoding='utf-8'))
    if ref.get('kind') != 'laya_probe_reference' or ref.get('backend') != 'cpu' or ref.get('module') != 'laya':
        raise ValueError('Reference must be captured unchanged CPU Laya')
    if ref.get('suite_sha256') != json_hash(suite) or [r['id'] for r in ref['records']] != [p['id'] for p in suite]:
        raise ValueError('Reference does not match suite')
    for probe, original in zip(suite, ref['records']):
        if not 1 <= len(probe['questions']) <= 4 or len(json.dumps(probe['state'], ensure_ascii=False)) > 16000:
            raise ValueError('Suite exceeds current HTTP schema; use CLI for five-question long validation')
        if original['input_sha256'] != json_hash([probe['state'], probe['questions']]):
            raise ValueError('Reference input identity differs')
    expected = None
    if args.expected_npu:
        expected = json.loads(args.expected_npu.read_text(encoding='utf-8'))
        if (expected.get('kind') != 'laya_probe_comparison' or expected.get('backend') != 'npu'
                or expected.get('suite_sha256') != json_hash(suite)
                or expected.get('reference_sha256') != sha256_file(args.reference)
                or expected.get('model_files') != ref.get('model_files')
                or not isinstance(expected.get('passed'), bool)):
            raise ValueError('Expected NPU report must be a completed same-suite comparison against this exact reference')
        if [r['id'] for r in expected['records']] != [p['id'] for p in suite]:
            raise ValueError('Expected NPU report is incomplete')
        buckets = sorted(expected.get('candidate_assets', {}).get('supported_buckets', []))
        if not buckets:
            raise ValueError('Expected NPU report has no supported buckets')
        total_calls, total_buckets = 0, {}
        for probe, original, candidate in zip(suite, ref['records'], expected['records']):
            if (candidate.get('input_sha256') != original['input_sha256']
                    or candidate.get('tokens') != original['tokens']
                    or candidate.get('tokens_equal') is not True
                    or candidate.get('reported_tokens_equal') is not True
                    or candidate.get('output', {}).get('usage') != original['output']['usage']):
                raise ValueError('Expected NPU report input/token identity differs from the reference')
            bucket = next((b for b in buckets if b >= max(t['tokens'] for t in original['tokens'].values())), None)
            calls = len(probe['questions'])
            expected_delta = {'npu_calls': calls, 'cpu_fallbacks': 0, 'bucket_calls': {str(bucket): calls}}
            if (bucket is None or candidate.get('npu_execution_verified') is not True
                    or candidate.get('accelerator_delta') != expected_delta
                    or candidate.get('expected_bucket_calls') != expected_delta['bucket_calls']):
                raise ValueError('Expected NPU report does not verify actual encoder execution')
            total_calls += calls
            total_buckets[str(bucket)] = total_buckets.get(str(bucket), 0) + calls
        stats = expected.get('accelerator_stats_after_measurement', {})
        if (stats.get('cpu_fallbacks') != 0 or stats.get('npu_calls') != total_calls
                or stats.get('bucket_calls') != total_buckets):
            raise ValueError('Expected NPU report aggregate execution counters are inconsistent')
    return suite, ref, expected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument('--suite', type=Path, default=ROOT/'tests/fidelity-probes.json')
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--expected-npu', type=Path,
                        help='Completed strict NPU comparison; required for integration_passed, independently of its fidelity gate')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--ready-timeout', type=float, default=300)
    args = parser.parse_args(argv)
    if args.output.exists() or args.timeout <= 0 or args.ready_timeout <= 0:
        parser.error('Use a fresh output and positive timeouts')
    suite, reference, expected = load_inputs(args)
    report = {'kind': 'laya_http_service_verification', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'base_url': args.base_url, 'suite_sha256': json_hash(suite),
              'reference_sha256': sha256_file(args.reference),
              'expected_npu_sha256': sha256_file(args.expected_npu) if args.expected_npu else None,
              'records': [], 'invalid_requests': [], 'passed': False, 'integration_passed': False,
              'runtime_checks_passed': False, 'development_fidelity_passed': None,
              'standalone_development_fidelity_passed': expected['passed'] if expected else None,
              'note': 'passed is an alias for integration_passed: HTTP validation, strict NPU execution, token usage, both bucket transitions, and exact standalone NPU output replay. It is not a fidelity qualification. development_fidelity_passed separately retains the unchanged <=5% decision-error and mean-TV <=5% gates on these development probes. Separate declared evaluations qualify fidelity. Token/marker hashes are established by the standalone runtime comparison.'}
    deadline = time.monotonic() + args.ready_timeout
    try:
        print('Waiting for strict NPU readiness', flush=True)
        while True:
            try:
                first = health(args.base_url, min(args.timeout, 10))
                break
            except (URLError, TimeoutError, ValueError) as error:
                if time.monotonic() >= deadline:
                    raise RuntimeError('Service readiness deadline expired') from error
                time.sleep(1)
        report['health_before'] = first
        invalid = [
            {'state': 'test', 'questions': {}},
            {'state': 'test', 'questions': {'q': {'type': 'choice', 'instructions': 'Choose.', 'criteria': ['one']}}},
            {'state': 'test', 'questions': {'q': {'type': 'score', 'instructions': 'Score.', 'criteria': {'a': 'A', 'b': 'B'}}}},
            {'state': 'x'*16001, 'questions': {'q': {'type': 'noul', 'instructions': 'Determine.'}}},
            {'state': 'test', 'questions': {str(i): {'type': 'noul', 'instructions': 'Determine.'} for i in range(5)}},
        ]
        for body in invalid:
            before = health(args.base_url, args.timeout)
            status, answer, elapsed = http_json(args.base_url, '/predict', body, args.timeout)
            after = health(args.base_url, args.timeout)
            changes = delta(before, after)
            passed = status == 422 and changes['npu_calls'] == 0 and not changes['bucket_calls']
            report['invalid_requests'].append({'http_status': status, 'elapsed_ms': elapsed,
                                                'accelerator_delta': changes, 'passed': passed, 'response': answer})
            if not passed:
                raise ValueError('Invalid HTTP input was not rejected before NPU inference')
        metrics, buckets_seen = [], []
        for position, (probe, original) in enumerate(zip(suite, reference['records'])):
            before = health(args.base_url, args.timeout)
            maximum = max(t['tokens'] for t in original['tokens'].values())
            bucket = next(b for b in sorted(before['supported_buckets']) if b >= maximum)
            status, answer, elapsed = http_json(args.base_url, '/predict',
                                               {k: probe[k] for k in ['state', 'questions']}, args.timeout)
            if status != 200:
                raise ValueError(f"HTTP{status} for {probe['id']}: {answer}")
            after = health(args.base_url, args.timeout)
            changes = delta(before, after)
            correct_delta = (changes['npu_calls'] == len(probe['questions']) and
                             changes['bucket_calls'] == {str(bucket): len(probe['questions'])} and
                             changes['cpu_fallbacks'] == 0 and changes['context_cache_errors'] == 0 and
                             changes['context_cache_misses'] == 0)
            values = {qid: decision_metrics(q, original['output']['answers'][qid], answer['answers'][qid])
                      for qid, q in probe['questions'].items()}
            metrics.extend(values.values())
            usage_equal = answer.get('usage') == original['output']['usage']
            matches_npu = None if expected is None else all(answer.get(k) == expected['records'][position]['output'].get(k) for k in ['answers', 'usage', 'model'])
            record = {'id': probe['id'], 'http_status': status, 'http_elapsed_ms': elapsed,
                      'api_elapsed_ms': answer.get('elapsed_ms'), 'expected_bucket': bucket,
                      'accelerator_delta': changes, 'npu_execution_verified': correct_delta,
                      'reported_tokens_equal': usage_equal, 'standalone_npu_outputs_equal': matches_npu,
                      'output': answer, 'decisions': values}
            report['records'].append(record)
            buckets_seen.append(bucket)
            print(probe['id'], 'bucket', bucket, 'calls', changes['npu_calls'], 'ms', round(elapsed, 1), flush=True)
            if not correct_delta or not usage_equal or matches_npu is False or answer.get('checkpoint') != after['checkpoint']:
                raise ValueError('HTTP response/counter/deployment equivalence check failed')
        transitions = list(zip(buckets_seen, buckets_seen[1:]))
        report['long_short_transition_verified'] = ((768, 1024) in transitions and (1024, 768) in transitions)
        report['overall'] = summarize_decisions(metrics)
        report['health_after'] = health(args.base_url, args.timeout)
        report['development_fidelity_passed'] = (report['overall']['decision_error_percent'] <= 5 and
                                                report['overall']['total_variation']['mean'] <= .05)
        report['runtime_checks_passed'] = report['long_short_transition_verified']
        report['integration_passed'] = (report['runtime_checks_passed'] and expected is not None and
                                        all(r['standalone_npu_outputs_equal'] is True for r in report['records']))
        report['passed'] = report['integration_passed']
        if expected is None:
            report['integration_not_verified_reason'] = 'Exact standalone NPU output replay requires --expected-npu'
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: report.get(k) for k in ['passed', 'integration_passed', 'development_fidelity_passed', 'overall', 'error']}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
