# 日本語Chain-of-Thoughtデータセットベンチマーク

このリポジトリは、[nlp-waseda/chain-of-thought-ja-dataset](https://github.com/nlp-waseda/chain-of-thought-ja-dataset)の日本語Chain-of-Thoughtデータセットを使用して、言語モデルのベンチマークを実施するためのツールを提供します。

## データセット概要

このデータセットには、以下の3つのカテゴリが含まれています：

1. **jcommonsenseqa**: 常識推論のデータセット（1,119問）
   - 選択肢から正解を選ぶ形式の問題
   - 例: 「Nintendoのゲームで動物が集まるのは？」

2. **last_letter_connection**: 記号推論のデータセット（1,000問）
   - 人名の姓と名の最後の文字を繋げる形式の問題
   - 例: 「金沢 京太郎」の苗字と名前それぞれの最後の文字を得て、それらを繋げてください。

3. **mawps**: 算数のデータセット（1,000問）
   - 文章題を解く形式の問題
   - 例: 「佐藤は16個の青い風船、鈴木は11個の青い風船、高橋は99個の青い風船を持っています。彼らは全部でいくつの青い風船を持っているのでしょう？」

各カテゴリには、テストデータとChain-of-Thoughtプロンプトのshotに用いる8問の問題と解答が含まれています。

## 必要な依存関係

このスクリプトを実行するには、以下の依存関係が必要です：

```bash
pip install requests
```

また、ローカルでOllamaが実行されている必要があります。Ollamaのインストール方法については、[Ollamaの公式ドキュメント](https://ollama.ai/)を参照してください。

## ベンチマークの実行方法

### 基本的な使い方

```bash
python benchmark.py <モデル名>
```

例：
```bash
python benchmark.py llama3
```

### オプション

- `--shot-type`: プロンプトのタイプを指定します（デフォルト: `shot`）
  - `shot`: Chain-of-Thoughtプロンプトを使用
  - `zero_shot`: 解答のみのプロンプトを使用

- `--dataset`: ベンチマークするデータセットを指定します（デフォルト: `all`）
  - `jcommonsenseqa`: 常識推論のデータセットのみ
  - `last_letter_connection`: 記号推論のデータセットのみ
  - `mawps`: 算数のデータセットのみ
  - `all`: すべてのデータセット

例：
```bash
# llama3モデルでzero-shotプロンプトを使用して常識推論のデータセットのみをベンチマーク
python benchmark.py llama3 --shot-type zero_shot --dataset jcommonsenseqa

# mistralモデルでChain-of-Thoughtプロンプトを使用してすべてのデータセットをベンチマーク
python benchmark.py mistral --shot-type shot --dataset all
```

### 注意事項

- モデル名に含まれるスラッシュやコロンは、アンダースコア（_）に置換されます。
  - 例: `meta/llama3` → `meta_llama3`
- ベンチマーク中は、10問ごとに進捗状況が保存されます。
- APIリクエストの間隔は0.5秒に設定されています。

## 結果の解釈

ベンチマーク結果は `results` ディレクトリに保存されます。各ファイルの命名規則は以下の通りです：

```
<サニタイズされたモデル名>_<データセット名>_<shot_type>.json
```

例：
```
llama3_jcommonsenseqa_shot.json
mistral_mawps_zero_shot.json
```

結果ファイルには以下の情報が含まれています：

- `dataset`: データセット名
- `shot_type`: プロンプトのタイプ
- `accuracy`: 正解率
- `correct`: 正解数
- `total`: 問題数
- `results`: 各問題の詳細結果
  - `question`: 問題文
  - `answer`: 正解
  - `predicted`: モデルの予測
  - `is_correct`: 正解かどうか

## 進捗状況の確認

ベンチマーク実行中は、10問ごとに進捗状況が保存されます。進捗状況ファイルの命名規則は以下の通りです：

```
<サニタイズされたモデル名>_<データセット名>_<shot_type>_progress.json
```

これにより、長時間のベンチマーク実行中に中断した場合でも、途中経過を確認できます。