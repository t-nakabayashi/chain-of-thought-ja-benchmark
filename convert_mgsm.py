import pandas as pd
import json
import random
from pathlib import Path

# TSVファイルを読み込む
df = pd.read_csv("dataset/mgsm/mgsm_ja.tsv", sep="\t", header=None, encoding="utf-8")
df.columns = ["question", "answer"]

# データをリストに変換
data = []
for _, row in df.iterrows():
    data.append({"question": row["question"], "answer": row["answer"]})

# データセットディレクトリを作成
Path("dataset/mgsm").mkdir(exist_ok=True)

# test.jsonを作成
with open("dataset/mgsm/test.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

# shot_example.jsonを作成（最初の8問を使用）
shot_examples = []
for i in range(min(8, len(data))):
    shot_examples.append({"shot_example": f"問題：{data[i]['question']}\n解答：答えは{data[i]['answer']}です。\n"})

with open("dataset/mgsm/shot_example.json", "w", encoding="utf-8") as f:
    json.dump(shot_examples, f, ensure_ascii=False, indent=4)

# zero_shot_example.jsonを作成（最初の8問を使用）
zero_shot_examples = []
for i in range(min(8, len(data))):
    zero_shot_examples.append({"shot_example": f"問題：{data[i]['question']}\n解答：答えは{data[i]['answer']}です。\n"})

with open("dataset/mgsm/zero_shot_example.json", "w", encoding="utf-8") as f:
    json.dump(zero_shot_examples, f, ensure_ascii=False, indent=4)

print(f"変換完了: {len(data)}問のデータを変換しました")
print(f"- test.json: {len(data)}問")
print(f"- shot_example.json: {len(shot_examples)}問")
print(f"- zero_shot_example.json: {len(zero_shot_examples)}問")
