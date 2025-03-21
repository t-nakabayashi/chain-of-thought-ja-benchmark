import pandas as pd

# TSVファイルを読み込む
df = pd.read_csv("dataset/mgsm/mgsm_ja.tsv", sep="\t", header=None, encoding="utf-8")

# 最初の5行を表示
print(df.head())

# 列数を確認
print(f"列数: {df.shape[1]}")
