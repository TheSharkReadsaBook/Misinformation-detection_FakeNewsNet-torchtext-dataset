
import re
import pandas as pd

INPUT_PATH = "../dataset/fakenewsnet-cleaned.json"
EXAMPLE_LIMIT = 3

df = pd.read_json(INPUT_PATH, lines=True)

# 명백한 웹페이지 코드 및 수집 흔적
PATTERNS = {
    "HTML 태그": r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*?)?\s*/?>",
    "Protected iframe": r"\[protected-iframe\b[^\]]*\]",
    "스크립트 코드": r"<script\b[^>]*>[\s\S]*?</script>",
    "스타일 코드": r"<style\b[^>]*>[\s\S]*?</style>",
    "HTML 엔티티": r"&(?:[A-Za-z][A-Za-z0-9]+|#\d+|#x[0-9a-fA-F]+);",
    "URL": r"https?://[^\s<>]+|www\.[^\s<>]+",
    "특수 공백": r"[\u00a0\u200b\u200c\u200d\ufeff]",
    "제어 문자": r"[\x00-\x08\x0b\x0c\x0e-\x1f]",
    "깨진 문자": r"\ufffd",
    "인코딩 오류 의심": r"â€™|â€œ|â€|Ã©|Ã±|Â\xa0",
}

# 문맥에 따라 정상적인 기사 내용일 수도 있으므로 검사만 수행
BOILERPLATE_PATTERNS = {
    "광고 안내": r"\bAdvertisement\b|\bAd Choices\b",
    "구독 안내": r"\bSubscribe to (?:our|the) newsletter\b",
    "관련 기사 안내": r"\bRelated Articles\b|\bRead More\b",
    "SNS 공유 안내": r"\bShare on (?:Facebook|Twitter|X)\b",
    "이미지 출처": r"\bGetty Images\b|\bShutterstock\b|\bAP Photo\b",
    "저작권 문구": r"\bAll rights reserved\b",
    "쿠키 안내": r"\bAccept Cookies\b|\bCookie Policy\b",
}

def check_patterns(column, patterns):
    series = df[column].fillna("").astype(str)

    for name, pattern in patterns.items():
        mask = series.str.contains(
            pattern,
            flags=re.IGNORECASE,
            regex=True
        )

        print(f"\n{name}: {mask.sum():,}건")

        for idx in series[mask].head(EXAMPLE_LIMIT).index:
            value = series.loc[idx]
            match = re.search(pattern, value, flags=re.IGNORECASE)

            if match:
                start = max(0, match.start() - 80)
                end = min(len(value), match.end() + 80)

                print(f"  행 번호: {idx + 1}")
                print(f"  발견 문자열: {match.group(0)[:200]!r}")
                print(f"  주변 문맥: {value[start:end]!r}")


def check_text_quality(column):
    series = df[column].fillna("").astype(str)

    lengths = series.str.len()
    words = series.str.split().str.len()

    print("\n[텍스트 품질]")
    print(f"빈 문자열: {(lengths == 0).sum():,}건")
    print(f"50자 미만: {(lengths < 50).sum():,}건")
    print(f"100자 미만: {(lengths < 100).sum():,}건")
    print(f"10단어 미만: {(words < 10).sum():,}건")

    # 문장 부호와 공백만으로 구성된 텍스트
    symbol_only = series.str.fullmatch(
        r"[\W_]*",
        na=False
    )

    print(f"기호/공백만 포함: {symbol_only.sum():,}건")

    for idx in series[symbol_only].head(EXAMPLE_LIMIT).index:
        print(f"  {idx + 1}행: {series.loc[idx][:200]!r}")


print("=" * 70)
print("FakeNewsNet 텍스트 품질 검사")
print("=" * 70)
print(f"전체 기사 수: {len(df):,}")

for column in ["title", "text"]:
    if column not in df.columns:
        continue

    print("\n" + "=" * 70)
    print(f"[{column}]")
    print("=" * 70)

    print("\n[1] 웹페이지 코드 및 특수문자")
    check_patterns(column, PATTERNS)

    print("\n[2] 웹페이지 잔여 문구 (제거 여부 미결정)")
    check_patterns(column, BOILERPLATE_PATTERNS)

    print("\n[3] 텍스트 길이 및 품질")
    check_text_quality(column)

print("\n" + "=" * 70)
print("검사 완료 - 원본 데이터 변경 없음")
print("=" * 70)




# ============================================================
# [추가] title, text 웹페이지 코드 및 특수문자 정제
# URL은 제거하지 않음
# ============================================================

import html

def clean_web_text(value):
    if not isinstance(value, str):
        return value

    # script, style 블록 제거
    value = re.sub(
        r"<(script|style)\b[^>]*>[\s\S]*?</\1\s*>",
        " ", value, flags=re.IGNORECASE
    )

    # Protected iframe 제거
    value = re.sub(
        r"\[protected-iframe\b[^\]]*\]",
        " ", value, flags=re.IGNORECASE
    )

    # HTML 태그 제거
    value = re.sub(
        r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*?)?\s*/?>",
        " ", value, flags=re.IGNORECASE
    )

    # HTML 엔티티 변환
    value = html.unescape(value)

    # 특수 공백 정리
    value = re.sub(r"[\u00a0\u200b\u200c\u200d\ufeff]", " ", value)

    # 제어 문자 제거
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", value)

    # 깨진 대체 문자 제거
    value = value.replace("\ufffd", "")

    # 연속 공백 정리
    value = re.sub(r"\s+", " ", value).strip()

    return value


print("\n" + "=" * 70)
print("[4] title, text 정제")
print("=" * 70)

df_clean = df.copy()

for column in ["title", "text"]:
    df_clean[column] = df_clean[column].apply(clean_web_text)

    changed = (df[column] != df_clean[column]).sum()
    print(f"{column} 변경된 기사 수: {changed:,}")


print("\n" + "=" * 70)
print("[5] 정제 후 웹페이지 코드 및 특수문자 재검사")
print("=" * 70)

# 남아 있는 HTML 엔티티 추가 변환
for column in ["title", "text"]:
    df_clean[column] = df_clean[column].apply(
        lambda value: html.unescape(value)
        if isinstance(value, str) else value
    )

for column in ["title", "text"]:
    print(f"\n[{column}]")

    series = df_clean[column].fillna("").astype(str)

    for name, pattern in PATTERNS.items():
        # URL은 검사 대상에서도 제외
        if name == "URL":
            continue

        count = series.str.contains(
            pattern,
            regex=True,
            flags=re.IGNORECASE
        ).sum()

        print(f"{name}: {count:,}건")


# 정제된 데이터를 새 JSONL 파일로 저장
output_path = "../dataset/fakenewsnet-text-cleaned.json"

df_clean.to_json(
    output_path,
    orient="records",
    lines=True,
    force_ascii=False
)

print("\n" + "=" * 70)
print("[6] 최종 결과")
print("=" * 70)
print(f"정제 전 기사 수: {len(df):,}")
print(f"정제 후 기사 수: {len(df_clean):,}")
print(f"저장 경로: {output_path}")

print("\n[7] 정제 후 남은 HTML 엔티티 확인")

pattern = PATTERNS["HTML 엔티티"]

for idx, value in df_clean["title"].items():
    matches = re.findall(pattern, str(value), flags=re.IGNORECASE)

    if matches:
        print(f"\n행 번호: {idx + 1}")
        print(f"제목: {value}")
        print(f"발견된 문자열: {matches}")