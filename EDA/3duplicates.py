import pandas as pd

df = pd.read_json("fakenewsnet-cleaned.json", lines=True)

print("=" * 60)
print("[1] 중복 제거 전")
print("=" * 60)

print(f"전체 기사 수: {len(df):,}")

# 제목과 본문이 동일한 모든 행 검사
duplicate_mask = df.duplicated(
    subset=["title", "text"],
    keep=False
)

duplicates = df[duplicate_mask]

# 라벨 충돌 그룹 검사
conflicts = duplicates.groupby(
    ["title", "text"],
    dropna=False
).filter(
    lambda group: group["label"].nunique(dropna=False) > 1
)

print(f"중복 그룹에 속한 전체 기사 수: {len(duplicates):,}")
print(f"라벨 충돌 그룹 수: {conflicts.groupby(['title', 'text'], dropna=False).ngroups:,}")
print(f"라벨 충돌 그룹에 속한 기사 수: {len(conflicts):,}")

# 중복 그룹의 모든 행 제거 (첫 번째 행도 제거)
df_clean = df[~duplicate_mask].copy()

print("\n" + "=" * 60)
print("[2] 중복 제거 후")
print("=" * 60)

print(f"제거 전 기사 수: {len(df):,}")
print(f"제거된 기사 수: {len(df) - len(df_clean):,}")
print(f"남은 기사 수: {len(df_clean):,}")

remaining = df_clean.duplicated(
    subset=["title", "text"],
    keep=False
).sum()

print(f"남은 중복 기사 수: {remaining:,}")

print("\n[3] 라벨 분포 비교")

print("\n제거 전:")
print(df["label"].value_counts())

print("\n제거 후:")
print(df_clean["label"].value_counts())

# [4] 중복 제거 결과를 새 JSON 파일로 저장
output_path = "fakenewsnet-deduplicated.json"

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