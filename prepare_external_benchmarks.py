#!/usr/bin/env python3
"""Prepare fixed external fidelity suites from pinned, data-only public sources.

This adapts BANKING77 to a fixed eight-intent subset, CLINC150 to ten domains,
and MASSIVE to eighteen scenarios. It is not a benchmark of native task accuracy.
Source labels are metadata only: every row in a suite receives identical options.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

from benchmark_fidelity import ROOT, json_hash, sha256_file

SEED = 'laya-external-2026-09-29-v1'
BANKING = {
    'card_payment_fee_charged': 'An unexpected fee was charged for a card payment.',
    'card_payment_not_recognised': 'A card payment is not recognized or was not authorized.',
    'card_payment_wrong_exchange_rate': 'The exchange rate used for a card payment is wrong.',
    'declined_card_payment': 'A card payment was declined or rejected.',
    'pending_card_payment': 'A card payment is still pending.',
    'request_refund': 'The customer wants to request a refund.',
    'reverted_card_payment?': 'A card payment was reversed or reverted.',
    'transaction_charged_twice': 'The same transaction was charged more than once.',
}
CLINC = {
    'auto_and_commute': 'Driving, vehicles, maintenance and commuting.',
    'banking': 'Bank accounts, balances, bills and money transfers.',
    'credit_cards': 'Credit cards, credit scores and card rewards.',
    'home': 'Home, music, shopping, calendars and personal lists.',
    'kitchen_and_dining': 'Food, cooking, restaurants and dining reservations.',
    'meta': 'Control or customize the assistant, settings and conversational responses.',
    'small_talk': 'Greetings, jokes and casual conversation about the assistant.',
    'travel': 'Flights, hotels, tourism and international travel information.',
    'utility': 'Weather, time, alarms, calculations, calls and messages.',
    'work': 'Employment, meetings, leave, salary and benefits.',
}
MASSIVE = {
    'alarm': 'Set or manage alarms.',
    'audio': 'Change volume or audio settings.',
    'calendar': 'Calendar events and reminders.',
    'cooking': 'Cooking and recipes.',
    'datetime': 'Time and dates.',
    'email': 'Read or send email.',
    'general': 'General assistant conversation or control.',
    'iot': 'Control smart home devices.',
    'lists': 'Manage lists.',
    'music': 'Music preferences or information.',
    'news': 'News updates.',
    'play': 'Play music, radio, podcasts or games.',
    'qa': 'Answer factual questions.',
    'recommendation': 'Recommend places, events or movies.',
    'social': 'Social media.',
    'takeaway': 'Order takeaway food.',
    'transport': 'Travel routes, taxis or tickets.',
    'weather': 'Weather forecasts.',
}


def ensure_sources(source_dir, lock):
    source_dir.mkdir(parents=True, exist_ok=True)
    for name, entry in lock.items():
        target = source_dir / name
        if not target.exists():
            if 'url' in entry:
                with urllib.request.urlopen(entry['url'], timeout=120) as stream:
                    data = stream.read()
            else:
                with tarfile.open(source_dir / 'massive-1.1.tar.gz') as archive:
                    data = archive.extractfile(entry['archive_member']).read()
            if hashlib.sha256(data).hexdigest() != entry['sha256']:
                raise ValueError(f'Download checksum mismatch: {name}')
            target.write_bytes(data)
        if sha256_file(target) != entry['sha256']:
            raise ValueError(f'Source checksum mismatch: {name}')


def rank(dataset, row_id):
    return hashlib.sha256(f'{SEED}|{dataset}|{row_id}'.encode()).hexdigest()


def probe(dataset, locale, row_id, state, intent, gold, semantic_id, criteria, instruction):
    return {
        'id': f'{dataset}-{locale}-{row_id}', 'state': state,
        'questions': {'route': {'type': 'choice', 'instructions': instruction,
                                'criteria': criteria}},
        'metadata': {'dataset': dataset, 'locale': locale, 'source_row_id': row_id,
                     'source_intent': intent, 'gold_label': gold, 'semantic_id': semantic_id},
    }


def build_suites(source_dir):
    suites = {}
    with (source_dir / 'banking-test.csv').open(encoding='utf-8', newline='') as stream:
        bank = list(csv.DictReader(stream))
    assert len(bank) == 3080
    suites['banking77-card8'] = [
        probe('banking77', 'en', str(i), r['text'], r['category'], r['category'],
              f'banking77-test-{i}', BANKING,
              'Which of these eight banking support intents best matches the customer request?')
        for i, r in enumerate(bank) if r['category'] in BANKING]
    assert all(sum(p['metadata']['gold_label'] == key for p in suites['banking77-card8']) == 40
               for key in BANKING)
    clinc = json.loads((source_dir / 'clinc-data_full.json').read_text(encoding='utf-8'))['test']
    domains = json.loads((source_dir / 'clinc-domains.json').read_text(encoding='utf-8'))
    assert len(clinc) == 4500 and set(domains) == set(CLINC)
    domain_by_intent = {intent: domain for domain, intents in domains.items() for intent in intents}
    selected = []
    for intent in sorted(domain_by_intent):
        rows = [i for i, row in enumerate(clinc) if row[1] == intent]
        assert len(rows) == 30
        selected.extend(sorted(rows, key=lambda i: rank('clinc150', str(i)))[:2])
    suites['clinc150-domain10'] = [
        probe('clinc150', 'en', str(i), clinc[i][0], clinc[i][1], domain_by_intent[clinc[i][1]],
              f'clinc150-test-{i}', CLINC, 'Which domain best matches this user request?')
        for i in sorted(selected)]
    by_locale = {}
    for locale in ['en-US', 'zh-TW', 'zh-CN']:
        rows = [json.loads(line) for line in (source_dir / f'massive-{locale}.jsonl').read_text(encoding='utf-8').splitlines()]
        by_locale[locale] = {str(r['id']): r for r in rows if r['partition'] == 'test'}
        assert len(by_locale[locale]) == 2974
    en = by_locale['en-US']
    assert set(r['scenario'] for r in en.values()) == set(MASSIVE)
    ids = []
    for scenario in MASSIVE:
        group = [i for i, r in en.items() if r['scenario'] == scenario]
        assert len(group) >= 10
        ids.extend(sorted(group, key=lambda i: rank('massive', i))[:10])
    for locale, rows in by_locale.items():
        for i in ids:
            assert rows[i]['scenario'] == en[i]['scenario'] and rows[i]['intent'] == en[i]['intent']
        suites[f'massive-{locale}'] = [
            probe('massive', locale, i, rows[i]['utt'], rows[i]['intent'], rows[i]['scenario'],
                  f'massive-test-{i}', MASSIVE, 'Which scenario best matches this user request?')
            for i in sorted(ids, key=int)]
    assert sum(map(len, suites.values())) == 1160
    return suites


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=ROOT / '.work/external-sources')
    parser.add_argument('--source-lock', type=Path, default=ROOT / 'reports/external-2026-09-29/sources.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports/external-2026-09-29')
    parser.add_argument('--plan', type=Path, default=ROOT / 'reports/evaluation-plans/external-datasets-2026-09-29.json')
    args = parser.parse_args()
    if args.plan.exists():
        parser.error('Use a fresh plan path; do not overwrite a frozen evaluation')
    lock = json.loads(args.source_lock.read_text(encoding='utf-8'))
    ensure_sources(args.source_dir, lock)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        'kind': 'laya_external_fidelity_plan', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'seed': SEED, 'source_lock': args.source_lock.resolve().relative_to(ROOT).as_posix(),
        'source_lock_sha256': sha256_file(args.source_lock),
        'criteria': {'decision_error_percent_max': 5.0, 'mean_total_variation_max': 0.05},
        'checkpoint_sha256': '9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204',
        'manifest_sha256': '95cadda20ef661ace20e2ba32ed41bef955d76219758cf6453806d8c32b986a7',
        'graph_sha256_by_bucket': {
            '768': '1bbade0d26358f6f3950b2928bb8e4344a170a240fed86e9ef50542c0ab5c138',
            '1024': 'cfd85315c4ee6b5bfaeb24ced44bba3cf6ebb37ff421efb4c65ade90c571ec64'},
        'selection': {
            'banking77-card8': 'All 40 official test rows in each of the eight fixed BANKING criteria labels (320).',
            'clinc150-domain10': 'Lowest two SHA256(seed|clinc150|zero-based-test-row) per each of 150 intents; mapped using official domains.json (300). No OOS rows.',
            'massive': 'Lowest ten SHA256(seed|massive|id) per each of 18 en-US test scenarios; exact same 180 IDs in en-US, zh-TW, zh-CN (540).'},
        'scope': 'Frozen external adapted choice-task fidelity; no model or calibration tuning on these results. Corpus training overlap of the original pretrained model is unknown. MASSIVE has 180 paired semantic IDs, not 540 independent semantic examples. Each suite must pass separately; small intent groups are diagnostic only. Gold-label accuracy is auxiliary and is not native BANKING77/CLINC150/MASSIVE leaderboard accuracy.',
        'suites': [],
    }
    for name, suite in build_suites(args.source_dir).items():
        path = args.output_dir / f'{name}-suite.json'
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(suite, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        prefix = args.output_dir.resolve().relative_to(ROOT).as_posix() + '/' + name
        plan['suites'].append({'name': name, 'cases': len(suite), 'suite': prefix + '-suite.json',
                               'reference': prefix + '-reference-wsl.json', 'candidate': prefix + '-npu.json',
                               'suite_sha256': json_hash(suite), 'suite_file_sha256': sha256_file(path)})
    with args.plan.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(plan, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'plan': str(args.plan), 'suites': [(s['name'], s['cases']) for s in plan['suites']]}, indent=2))


if __name__ == '__main__':
    main()
