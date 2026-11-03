
import json
import pandas as pd

file_path = "fakenewsnet-cleaned.json"

# JSON / JSONL 자동 읽기
try:
    df = pd.read_json(file_path, lines=True)
except ValueError:
    df = pd.read_json(file_path)

# 처음 10행 출력
for i, (_, row) in enumerate(df.head(10).iterrows(), 1):
    print("\n" + "=" * 70)
    print(f"기사 {i}")
    print("=" * 70)

    for column, value in row.items():
        print(f"\n[{column}]")

        if isinstance(value, (dict, list)):
            print(json.dumps(value, ensure_ascii=False, indent=2))

        elif column == "text" and isinstance(value, str):
            print(value[:200] + ("..." if len(value) > 200 else ""))

        else:
            print(value)

print("\n총 출력 기사 수:", min(10, len(df)))
