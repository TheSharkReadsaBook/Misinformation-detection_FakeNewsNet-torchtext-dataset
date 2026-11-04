# 01 EDA 시작하기: 크기, 자료형, 결측
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

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


def main():
    parser = argparse.ArgumentParser(
        description="FakeNewsNet EDA: 크기, 자료형, 결측치 확인"
    )

    parser.add_argument(
        "file",
        nargs="?",
        default="../dataset/fakenewsnet-unbalanced.json"
        #default="dataset/fakenewsnet-cleaned.json"
        # default="dataset/fakenewsnet-deduplicated.json"
    )

    args = parser.parse_args()

    records = load_records(Path(args.file))

    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                f"{i + 1}번째 레코드가 JSON 객체가 아닙니다."
            )

    df = pd.DataFrame(records)
    total = len(records)

    # JSON에 실제 존재하는 변수만 사용
    fields = df.columns.tolist()

    print("=" * 85)
    print("01 EDA 시작하기: 크기, 자료형, 결측")
    print("=" * 85)

    # 1. 데이터 크기
    print("\n[1] 데이터 크기 (df.shape)")
    print(df.shape)
    print(f"행 개수: {df.shape[0]:,}")
    print(f"열 개수: {df.shape[1]:,}")

    # 2. 변수 목록
    print("\n[2] JSON에 존재하는 변수")
    for i, field in enumerate(fields, 1):
        print(f"{i}. {field}")

    # 3. 자료형
    print("\n[3] 자료형 (df.dtypes)")
    print(df.dtypes.to_string())

    # 4. 자료형별 열 개수
    print("\n[4] 자료형별 열 개수")
    print(
        df.dtypes.astype(str)
        .value_counts()
        .to_string()
    )

    # 5. 결측치
    stats = defaultdict(Counter)

    for record in records:
        for field in fields:
            kind = missing_kind(record, field)

            if kind:
                stats[field][kind] += 1

    print("\n[5] 결측 개수와 비율")
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

    print("\n참고:")
    print("- date=0은 결측치로 처리")
    print("- spread/user의 0은 결측치로 처리하지 않음")
    print("- 빈 문자열은 결측치로 처리")
    print("- CSV 및 JSON 파일 저장 없음")


if __name__ == "__main__":
    main()
