# political fact와 gossip title 매칭하여 구분 1:1로 매칭된 것만 출력함
#!/usr/bin/env python3
"""Match cleaned FakeNewsNet records to original GossipCop/PolitiFact CSV titles.

Requires: pandas (pip install pandas)
Usage: python match_fakenewsnet_sources.py fakenewsnet-unbalanced.json
"""
import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

BASE = 'https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/master/dataset/'
FILES = {
    'gossipcop_fake.csv': ('GossipCop', 'fake'),
    'gossipcop_real.csv': ('GossipCop', 'real'),
    'politifact_fake.csv': ('PolitiFact', 'fake'),
    'politifact_real.csv': ('PolitiFact', 'real'),
}


def norm_title(value):
    if not isinstance(value, str):
        return ''
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def norm_label(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        v = value.strip().casefold()
        if v in ('fake', 'false'):
            return 'fake'
        if v in ('real', 'true'):
            return 'real'
        return None
    return None


def load_records(path):
    """Read a JSON array/object or newline-delimited JSON (JSONL/NDJSON)."""
    with path.open('r', encoding='utf-8-sig') as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError as exc:
            # "Extra data" commonly indicates consecutive JSON objects, one per line.
            if exc.msg != 'Extra data':
                raise ValueError(f'Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}') from exc
            f.seek(0)
            rows = []
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as line_exc:
                    raise ValueError(
                        f'Not valid JSONL: line {line_no}, column {line_exc.colno}: '
                        f'{line_exc.msg}. Check whether records span multiple lines.'
                    ) from line_exc
                if not isinstance(row, dict):
                    raise ValueError(f'JSONL line {line_no} must be an object')
                rows.append(row)
            return rows, 'jsonl'
    if isinstance(obj, list):
        return obj, None
    if isinstance(obj, dict):
        if all(isinstance(v, list) for v in obj.values()):
            rows = []
            for group, group_rows in obj.items():
                for item in group_rows:
                    if not isinstance(item, dict):
                        raise ValueError('JSON group contains non-object records')
                    record = dict(item)
                    record.setdefault('label', group)
                    rows.append(record)
            return rows, 'grouped'
        for key in ('data', 'articles', 'records'):
            if isinstance(obj.get(key), list):
                return obj[key], key
    raise ValueError('Unsupported JSON layout: expected a list, grouped records, or JSONL')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('json_file', type=Path)
    parser.add_argument('--csv-dir', type=Path, default=Path('original_csv'), help='Local CSV cache directory')
    parser.add_argument('--output-dir', type=Path, default=Path('matched_output'))
    parser.add_argument('--numeric-labels', choices=['zero-fake', 'one-fake'], default=None,
                        help='Specify only if cleaned JSON uses numeric 0/1 labels')
    args = parser.parse_args()

    records, layout = load_records(args.json_file)
    if not all(isinstance(row, dict) for row in records):
        raise ValueError('Each JSON record must be an object')
    args.csv_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference = {}
    for filename, (source, label) in FILES.items():
        local = args.csv_dir / filename
        if not local.exists():
            import urllib.request
            print(f'Downloading {filename} ...')
            urllib.request.urlretrieve(BASE + filename, local)
        df = pd.read_csv(local, dtype=str, keep_default_na=False)
        if 'title' not in df.columns:
            raise ValueError(f'{filename}: title column missing')
        for title in df['title']:
            key = (norm_title(title), label)
            if key[0]:
                reference.setdefault(key, set()).add(source)

    def clean_label(value):
        normalized = norm_label(value)
        if normalized is not None:
            return normalized
        if args.numeric_labels and not isinstance(value, bool):
            s = str(value).strip()
            if s in ('0', '1'):
                return ('fake' if s == '0' else 'real') if args.numeric_labels == 'zero-fake' else ('fake' if s == '1' else 'real')
        return None

    counts = Counter()
    results = []
    for record in records:
        item = dict(record)
        title = norm_title(item.get('title'))
        label = clean_label(item.get('label'))
        sources = reference.get((title, label), set()) if title and label else set()
        if len(sources) == 1:
            source = next(iter(sources))
            status = 'matched_unique'
        elif len(sources) > 1:
            source = 'unknown'
            status = 'ambiguous_source'
        elif not title:
            source = 'unknown'
            status = 'missing_title'
        elif not label:
            source = 'unknown'
            status = 'unrecognized_label'
        else:
            source = 'unknown'
            status = 'no_exact_match'
        item['source'] = source
        item['source_match_status'] = status
        results.append(item)
        counts[(source, status)] += 1

    json_path = args.output_dir / 'fakenewsnet_with_source.json'
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    # CSV may contain nested data; encode it as JSON strings.
    flat = []
    for item in results:
        flat.append({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                     for k, v in item.items()})
    csv_path = args.output_dir / 'fakenewsnet_with_source.csv'
    pd.DataFrame(flat).to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f'\nTotal: {len(results):,}')
    for source in ('GossipCop', 'PolitiFact', 'unknown'):
        print(f'{source}: {sum(n for (s, _), n in counts.items() if s == source):,}')
    print('\nMatch status:')
    for status, count in sorted(Counter({st: sum(n for (s, t), n in counts.items() if t == st)
                                         for _, st in counts}).items()):
        print(f'  {status}: {count:,}')
    print(f'\nSaved: {json_path} / {csv_path}')
    print('Note: title+label exact matches are evidence-based, but not a proof of record identity.')


if __name__ == '__main__':
    main()
