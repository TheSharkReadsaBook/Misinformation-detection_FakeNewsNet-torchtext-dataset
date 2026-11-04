
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

DROP_FIELDS = {"summary", "keywords"}
MISSING_STRINGS = {"null", "none", "nan", "na", "n/a"}


def load_records(path):
    """일반 JSON 배열과 JSONL 형식 모두 지원"""
    with path.open("r", encoding="utf-8-sig") as f:
        content = f.read()

    try:
        data = json.loads(content)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            for key in ("data", "articles", "records"):
                if isinstance(data.get(key), list):
                    return data[key]
            return [data]

    except json.JSONDecodeError:
        pass

    records = []

    for lineno, line in enumerate(content.splitlines(), 1):
        if not line.strip():
            continue

        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"JSONL {lineno}번째 줄 파싱 오류: {e}"
            ) from e

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

    if field == "date":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value == 0:
                return "date=0"

        elif isinstance(value, str) and value.strip() in {"0", "0.0"}:
            return "date=0"

    return None


def missing_summary(records, fields):
    stats = defaultdict(Counter)

    for record in records:
        for field in fields:
            kind = missing_kind(record, field)

            if kind:
                stats[field][kind] += 1

    return stats


def print_missing_summary(records, fields, stats, heading):
    total = len(records)

    print(f"\n{heading}")
    print(f"전체 기사 수: {total:,}\n")

    print(
        f"{'필드명':<18} "
        f"{'결측 개수':>10} "
        f"{'결측률(%)':>12}  "
        "결측 유형별 개수"
    )
    print("-" * 85)

    for field in fields:
        n = sum(stats[field].values())
        rate = n / total * 100 if total else 0

        details = ", ".join(
            f"{kind}: {count:,}"
            for kind, count in sorted(stats[field].items())
        ) or "없음"

        print(
            f"{field:<18} "
            f"{n:>10,} "
            f"{rate:>11.2f}%  "
            f"{details}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="FakeNewsNet EDA: 결측 확인 및 결측 행 제거"
    )

    parser.add_argument(
        "file",
        nargs="?",
        default="../dataset/fakenewsnet-unbalanced.json"
    )

    parser.add_argument(
        "--clean-output",
        default="../dataset/fakenewsnet-cleaned.json"
    )

    args = parser.parse_args()

    records = load_records(Path(args.file))

    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                f"{i + 1}번째 레코드가 JSON 객체가 아닙니다."
            )

    # 1. summary, keywords 변수 전체 삭제
    records = [
        {
            k: v
            for k, v in record.items()
            if k not in DROP_FIELDS
        }
        for record in records
    ]

    # 2. JSON에 실제 존재하는 변수 자동 탐색
    df = pd.DataFrame(records)
    fields = df.columns.tolist()

    print("=" * 85)
    print("FakeNewsNet EDA: 크기, 자료형, 결측 확인 및 제거")
    print("=" * 85)

    print("\n제외한 변수: summary, keywords")

    print("\n[1] 변수 제거 후 데이터 크기")
    print(f"행 개수: {df.shape[0]:,}")
    print(f"열 개수: {df.shape[1]:,}")

    print("\n[2] JSON에 존재하는 변수")
    for i, field in enumerate(fields, 1):
        print(f"{i}. {field}")

    print("\n[3] 자료형")
    print(df.dtypes.to_string())

    print("\n[4] 자료형별 열 개수")
    print(
        df.dtypes.astype(str)
        .value_counts()
        .to_string()
    )

    # 3. 결측 제거 전 현황
    before_stats = missing_summary(records, fields)

    print_missing_summary(
        records,
        fields,
        before_stats,
        "[5] 제거 전 결측 현황"
    )

    # 4. 하나라도 결측이 있는 행 제거
    clean_records = []
    removed_by_field = Counter()

    for record in records:
        missing_fields = [
            field
            for field in fields
            if missing_kind(record, field)
        ]

        if missing_fields:
            removed_by_field.update(missing_fields)
        else:
            clean_records.append(record)

    removed = len(records) - len(clean_records)
    rate = removed / len(records) * 100 if records else 0

    print("\n[6] 결측 행 제거 결과")
    print(f"원본 기사 수: {len(records):,}")
    print(f"제거 기사 수: {removed:,} ({rate:.2f}%)")
    print(f"남은 기사 수: {len(clean_records):,}")

    print("\n결측 필드별 제거 대상 기사 수 (중복 집계 가능):")

    for field, count in removed_by_field.most_common():
        print(f"  {field}: {count:,}")

    # 5. 결측 제거 후 현황
    after_stats = missing_summary(clean_records, fields)

    print_missing_summary(
        clean_records,
        fields,
        after_stats,
        "[7] 제거 후 결측 현황"
    )

    # 6. 정제된 JSONL 저장
    clean_output = Path(args.clean_output)

    with clean_output.open("w", encoding="utf-8") as f:
        for record in clean_records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False
                ) + "\n"
            )

    print("\n[8] 저장 결과")
    print(f"결측 제거 JSONL: {clean_output.resolve()}")

    print("\n참고:")
    print("- summary, keywords 변수 전체 삭제")
    print("- JSON에 존재하는 나머지 모든 변수 자동 탐색")
    print("- 하나라도 결측이면 해당 행 삭제")
    print("- date=0은 결측으로 처리")
    print("- spread/user=0은 정상값으로 유지")
    print("- CSV 파일 저장 없음")


if __name__ == "__main__":
    main()
