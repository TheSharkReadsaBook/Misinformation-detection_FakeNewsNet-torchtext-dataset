import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_FIELDS = ["title", "text", "tweets", "spread", "user", "date", "summary", "keywords", "label"]
MISSING_STRINGS = {"null", "none", "nan", "na", "n/a"}


def load_records(path):
    with path.open("r", encoding="utf-8-sig") as f:
        first = f.read(1)
        while first and first.isspace():
            first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON 배열이 아닙니다.")
            return data
        if first == "{":
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                if e.msg != "Extra data":
                    raise
            else:
                if isinstance(data, dict):
                    for key in ("data", "records", "articles", "items"):
                        if isinstance(data.get(key), list):
                            return data[key]
                    return [data]
                if isinstance(data, list):
                    return data
        records = []
        f.seek(0)
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL {lineno}번째 줄 파싱 오류: {e}") from e
        return records


def missing_kind(record, field):
    if field not in record:
        return "키 없음"
    value = record[field]
    if value is None:
        return "null"
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return "빈 문자열/공백"
        if stripped.lower() in MISSING_STRINGS:
            return "결측 표현 문자열"
    if isinstance(value, (list, dict)) and not value:
        return "빈 리스트/객체"
    if field == "date" and not isinstance(value, bool) and isinstance(value, (int, float)) and value == 0:
        return "date=0"
    if field == "date" and isinstance(value, str) and value.strip() in {"0", "0.0"}:
        return "date=0"
    return None


def main():
    parser = argparse.ArgumentParser(description="FakeNewsNet 정제본 필드별 결측치 검사")
    parser.add_argument("file", nargs="?", default="fakenewsnet-unbalanced.json")
    parser.add_argument("--output", default="missing_report.csv")
    args = parser.parse_args()
    records = load_records(Path(args.file))
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{i + 1}번째 레코드가 JSON 객체가 아닙니다.")
    total = len(records)
    fields = list(EXPECTED_FIELDS)
    fields.extend(sorted(set().union(*(r.keys() for r in records)) - set(fields)) if records else [])
    stats = defaultdict(Counter)
    for record in records:
        for field in fields:
            kind = missing_kind(record, field)
            if kind:
                stats[field][kind] += 1
    print(f"전체 기사 수: {total:,}\n")
    print(f"{'필드명':<18} {'결측 개수':>10} {'결측률(%)':>12}  결측 유형별 개수")
    print("-" * 85)
    rows = []
    for field in fields:
        n = sum(stats[field].values())
        rate = n / total * 100 if total else 0
        details = ", ".join(f"{k}: {v:,}" for k, v in sorted(stats[field].items())) or "없음"
        print(f"{field:<18} {n:>10,} {rate:>11.2f}%  {details}")
        rows.append((field, n, f"{rate:.2f}", details))
    import csv
    output = Path(args.output)
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["필드명", "결측 개수", "결측률(%)", "결측 유형별 개수"])
        writer.writerows(rows)
    print(f"\n결과 저장: {output.resolve()}")
    print("참고: date=0은 결측 처리, spread/user의 0은 정상값으로 처리합니다.")


if __name__ == "__main__":
    main()
