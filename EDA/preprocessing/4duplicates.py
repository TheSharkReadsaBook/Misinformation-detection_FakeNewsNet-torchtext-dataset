import pandas as pd

df = pd.read_json("../dataset/fakenewsnet-text-cleaned.json", lines=True)

print("=" * 60)
print("[1] 중복 제거 전")
print("=" * 60)

print(f"전체 기사 수: {len(df):,}")

# 제목 중복 검사 (확인용, 제거 기준에는 포함하지 않음)
title_mask = df.duplicated(
    subset=["title"],
    keep=False
)

# 본문 중복 검사
text_mask = df.duplicated(
    subset=["text"],
    keep=False
)

# 제목과 본문이 모두 동일한 경우
both_mask = df.duplicated(
    subset=["title", "text"],
    keep=False
)

# 본문이 동일하면 제목과 관계없이 제거 대상
duplicate_mask = text_mask

duplicates = df[duplicate_mask]

# 본문이 동일한 그룹에서 라벨 충돌 검사
conflicts = duplicates.groupby(
    ["text"],
    dropna=False
).filter(
    lambda group: group["label"].nunique(dropna=False) > 1
)

print(f"제목 중복 기사 수: {title_mask.sum():,}")
print(f"본문 중복 기사 수: {text_mask.sum():,}")
print(f"제목 + 본문 중복 기사 수: {both_mask.sum():,}")
print(f"중복 제거 대상 전체 기사 수: {len(duplicates):,}")

print(
    f"라벨 충돌 그룹 수: "
    f"{conflicts.groupby(['text'], dropna=False).ngroups:,}"
)
print(f"라벨 충돌 그룹에 속한 기사 수: {len(conflicts):,}")

# 본문이 중복된 모든 행 제거 (첫 번째 행도 제거)
df_clean = df[~duplicate_mask].copy()

print("\n" + "=" * 60)
print("[2] 중복 제거 후")
print("=" * 60)

print(f"제거 전 기사 수: {len(df):,}")
print(f"제거된 기사 수: {len(df) - len(df_clean):,}")
print(f"남은 기사 수: {len(df_clean):,}")

# 중복 재검사
remaining_title = df_clean.duplicated(
    subset=["title"],
    keep=False
).sum()

remaining_text = df_clean.duplicated(
    subset=["text"],
    keep=False
).sum()

remaining_both = df_clean.duplicated(
    subset=["title", "text"],
    keep=False
).sum()

print(f"남은 제목 중복 기사 수: {remaining_title:,}")
print(f"남은 본문 중복 기사 수: {remaining_text:,}")
print(f"남은 제목 + 본문 중복 기사 수: {remaining_both:,}")

print("\n[3] 라벨 분포 비교")

print("\n제거 전:")
print(df["label"].value_counts())

print("\n제거 후:")
print(df_clean["label"].value_counts())

# [4] 중복 제거 결과를 새 JSON 파일로 저장
output_path = "../dataset/fakenewsnet-deduplicated.json"

df_clean.to_json(
    output_path,
    orient="records",
    lines=True,
    force_ascii=False,
    indent=None
)

print("\n[4] JSON 파일 저장")
print(f"저장 경로: {output_path}")
print(f"저장된 기사 수: {len(df_clean):,}")