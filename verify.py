import json
import statistics
import time
import urllib.request
import urllib.error
from pathlib import Path
ROOT = Path(__file__).resolve().parent
BASE = 'http://127.0.0.1:8000'

def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as res:
        return json.load(res)

for i in range(90):
    try:
        health = request('/health')
        break
    except (urllib.error.URLError, TimeoutError):
        time.sleep(1)
else:
    raise RuntimeError('Service failed to start')

example = json.loads((ROOT / 'example.json').read_text())
results = []
for i in range(4):
    result = request('/predict', example)
    assert result['answers']['department']['choice'] == 'billing', result
    assert result['answers']['refund_requested']['noul'] > 0.5, result
    results.append(result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
negative = request('/predict', {'state': '我只是想查詢新方案的價格，不需要退款。', 'questions': {'refund_requested': example['questions']['refund_requested']}})
negative_passed = negative['answers']['refund_requested']['noul'] < 0.5
assert 0 <= negative['answers']['refund_requested']['noul'] <= 1
score = request('/predict', {'state': 'The entire production system is down. All customers are blocked.', 'questions': {'urgency': {'type': 'score', 'instructions': 'How urgent is this issue?', 'criteria': ['not urgent', 'soon', 'critical deadline or blocking issue']}}})
assert 0 <= score['answers']['urgency']['score'] <= 2, score
for invalid in ({'state':'x','questions':{}}, {'state':'x','questions':{'x':{'type':'choice','instructions':'pick','criteria':[]}}}):
    try:
        request('/predict', invalid)
        raise AssertionError('Invalid request accepted')
    except urllib.error.HTTPError as exc:
        assert exc.code == 422, exc.code
report = {'health': health, 'cold_request_ms': results[0]['elapsed_ms'], 'warm_two_questions_median_ms': statistics.median(r['elapsed_ms'] for r in results[1:]), 'positive': results[-1], 'negative': negative, 'negative_semantic_check_passed': negative_passed, 'score': score, 'invalid_requests': '422 as expected'}
(ROOT / 'verification.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
print(json.dumps(report, indent=2, ensure_ascii=False))
