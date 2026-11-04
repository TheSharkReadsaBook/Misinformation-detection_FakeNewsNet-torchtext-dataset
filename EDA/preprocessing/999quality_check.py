
import re
from collections import Counter

import pandas as pd


INPUT_PATH = "../dataset/fakenewsnet-deduplicated.json"
EXAMPLE_LIMIT = 5

df = pd.read_json(INPUT_PATH, lines=True)

COLUMNS = ["title", "text"]

for col in COLUMNS + ["label"]:
    if col not in df.columns:
        raise ValueError(f"필수 컬럼 없음: {col}")

print("=" * 75)
print("FakeNewsNet 추가 데이터 품질 검사")
print("=" * 75)
print(f"전체 기사 수: {len(df):,}")


def show_examples(mask, column, limit=EXAMPLE_LIMIT):
    """문제가 발견된 실제 기사 사례 출력"""
    for idx in df.index[mask][:limit]:
        value = str(df.at[idx, column])
        print(f"  {idx + 1}행: {value[:300]!r}")


def inspect_pattern(column, name, pattern):
    """패턴에 해당하는 기사 수 및 발견 문자열 출력"""
    series = df[column].fillna("").astype(str)

    mask = series.str.contains(
        pattern,
        regex=True,
        flags=re.IGNORECASE,
        na=False
    )

    print(f"\n{name}: {mask.sum():,}건")

    for idx in df.index[mask][:EXAMPLE_LIMIT]:
        value = series.loc[idx]
        match = re.search(pattern, value, re.IGNORECASE)

        if match:
            start = max(0, match.start() - 60)
            end = min(len(value), match.end() + 60)

            print(f"  {idx + 1}행")
            print(f"  발견: {match.group(0)[:150]!r}")
            print(f"  문맥: {value[start:end]!r}")


# ============================================================
# 1. 정제 후 중복 검사
# ============================================================

print("\n" + "=" * 75)
print("[1] 정제 후 중복 검사")
print("=" * 75)

for name, columns in [
    ("제목 중복", ["title"]),
    ("본문 중복", ["text"]),
    ("제목 + 본문 중복", ["title", "text"])
]:
    mask = df.duplicated(subset=columns, keep=False)

    groups = df.loc[mask].groupby(
        columns,
        dropna=False
    ).ngroups

    print(f"{name}")
    print(f"  중복 그룹 수: {groups:,}")
    print(f"  중복 그룹에 속한 기사 수: {mask.sum():,}")


# ============================================================
# 2. 빈 문자열 검사
# ============================================================

print("\n" + "=" * 75)
print("[2] 빈 제목 및 본문")
print("=" * 75)

for column in COLUMNS:
    series = df[column].fillna("").astype(str)

    mask = series.str.strip().eq("")

    print(f"\n{column} 빈 문자열: {mask.sum():,}건")
    show_examples(mask, column)


# ============================================================
# 3. 텍스트 길이 검사
# ============================================================

print("\n" + "=" * 75)
print("[3] 제목 및 본문 길이")
print("=" * 75)

for column in COLUMNS:
    series = df[column].fillna("").astype(str)

    char_lengths = series.str.len()
    word_lengths = series.str.split().str.len()

    print(f"\n[{column}]")
    print(f"평균 글자 수: {char_lengths.mean():.1f}")
    print(f"중앙값 글자 수: {char_lengths.median():.1f}")
    print(f"최소 글자 수: {char_lengths.min()}")
    print(f"최대 글자 수: {char_lengths.max()}")
    print(f"평균 단어 수: {word_lengths.mean():.1f}")

    if column == "text":
        for threshold in [50, 100, 200, 500]:
            mask = char_lengths < threshold

            print(f"\n{threshold}자 미만: {mask.sum():,}건")
            show_examples(mask, column, limit=3)


# ============================================================
# 4. 웹페이지 코드 및 특수문자 잔여물
# ============================================================

print("\n" + "=" * 75)
print("[4] HTML 및 웹페이지 코드 잔여물")
print("=" * 75)

WEB_PATTERNS = {
    "HTML 태그": r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*?)?\s*/?>",
    "Protected iframe": r"\[protected-iframe\b[^\]]*\]",
    "script 태그": r"<script\b",
    "style 태그": r"<style\b",
    "HTML 엔티티": r"&(?:[A-Za-z][A-Za-z0-9]+|#\d+|#x[0-9a-fA-F]+);",
    "특수 공백": r"[\u00a0\u200b\u200c\u200d\ufeff]",
    "제어 문자": r"[\x00-\x08\x0b\x0c\x0e-\x1f]",
    "깨진 문자": r"\ufffd",
    "인코딩 오류 의심": r"â€™|â€œ|â€|Ã©|Ã±|Â\xa0",
    "URL": r"https?://[^\s<>]+|www\.[^\s<>]+",
}

for column in COLUMNS:
    print(f"\n[{column}]")

    for name, pattern in WEB_PATTERNS.items():
        inspect_pattern(column, name, pattern)


# ============================================================
# 5. 웹페이지 잔여 문구 검사
# ============================================================

print("\n" + "=" * 75)
print("[5] 광고·구독·공유·관련 기사 문구")
print("=" * 75)

BOILERPLATE_PATTERNS = {
    "광고 문구": r"\bAdvertisement\b|\bAd Choices\b",
    "구독 안내": r"\bSubscribe to (?:our|the) newsletter\b",
    "SNS 공유": r"\bShare on (?:Facebook|Twitter|X)\b|\bTweet this\b",
    "관련 기사": r"\bRelated Articles\b|\bRead More\b",
    "이미지 출처": r"\bGetty Images\b|\bShutterstock\b|\bAP Photo\b",
    "저작권": r"\bAll rights reserved\b|©\s*\d{4}",
    "쿠키 안내": r"\bAccept Cookies\b|\bCookie Policy\b",
    "개인정보 안내": r"\bPrivacy Policy\b|\bTerms of Service\b",
}

for column in COLUMNS:
    print(f"\n[{column}]")

    for name, pattern in BOILERPLATE_PATTERNS.items():
        inspect_pattern(column, name, pattern)


# ============================================================
# 6. 본문 내부 반복 문단 검사
# ============================================================

print("\n" + "=" * 75)
print("[6] 본문 내부 반복 문단")
print("=" * 75)


def find_repeated_paragraphs(value):
    if not isinstance(value, str):
        return []

    # 줄바꿈으로 구분된 문단만 검사
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n+", value)
        if len(p.strip()) >= 80
    ]

    counts = Counter(paragraphs)

    return [
        (paragraph, count)
        for paragraph, count in counts.items()
        if count > 1
    ]


repeat_count = 0

for idx, value in df["text"].items():
    repeats = find_repeated_paragraphs(value)

    if repeats:
        repeat_count += 1

        if repeat_count <= EXAMPLE_LIMIT:
            print(f"\n{idx + 1}행")
            for paragraph, count in repeats[:3]:
                print(f"  반복 횟수: {count}")
                print(f"  문단: {paragraph[:200]!r}")

print(f"\n반복 문단이 발견된 기사 수: {repeat_count:,}")
print("참고: 줄바꿈이 없는 본문에서는 문단 반복을 탐지할 수 없음")


# ============================================================
# 7. 기호 위주 텍스트 검사
# ============================================================

print("\n" + "=" * 75)
print("[7] 기호 위주 텍스트")
print("=" * 75)

for column in COLUMNS:
    series = df[column].fillna("").astype(str)

    # 영문자와 숫자가 하나도 없는 경우
    mask = ~series.str.contains(r"[A-Za-z0-9]", regex=True)

    print(f"\n{column}: {mask.sum():,}건")
    show_examples(mask, column)


# ============================================================
# 8. 라벨 분포
# ============================================================

print("\n" + "=" * 75)
print("[8] 라벨 분포")
print("=" * 75)

counts = df["label"].value_counts(dropna=False)
ratios = df["label"].value_counts(
    normalize=True,
    dropna=False
) * 100

for label, count in counts.items():
    print(f"{label}: {count:,}건 ({ratios[label]:.2f}%)")


# ============================================================
# 9. 최종 결과
# ============================================================

print("\n" + "=" * 75)
print("[9] 검사 완료")
print("=" * 75)

print(f"전체 기사 수: {len(df):,}")
print("검사 대상: title, text")
print("데이터 수정 없음")
print("파일 저장 없음")
