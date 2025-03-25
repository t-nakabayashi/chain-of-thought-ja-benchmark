# 日本語Chain-of-Thoughtデータセットベンチマーク

このリポジトリは、[nlp-waseda/chain-of-thought-ja-dataset](https://github.com/nlp-waseda/chain-of-thought-ja-dataset)の日本語Chain-of-Thoughtデータセットを使用して、言語モデルのベンチマークを実施するためのツールを提供します。

## データセット概要

このデータセットには、以下の4つのカテゴリが含まれています：

1. **jcommonsenseqa**: 常識推論のデータセット（1,119問）
   - 選択肢から正解を選ぶ形式の問題
   - 例: 「Nintendoのゲームで動物が集まるのは？」

2. **last_letter_connection**: 記号推論のデータセット（1,000問）
   - 人名の姓と名の最後の文字を繋げる形式の問題
   - 例: 「金沢 京太郎」の苗字と名前それぞれの最後の文字を得て、それらを繋げてください。

3. **mawps**: 算数のデータセット（1,000問）
   - 文章題を解く形式の問題
   - 例: 「佐藤は16個の青い風船、鈴木は11個の青い風船、高橋は99個の青い風船を持っています。彼らは全部でいくつの青い風船を持っているのでしょう？」

4. **mgsm**: 多言語の算数文章題データセット（250問）
   - 複雑な算数の文章題を解く形式の問題
   - 例: 「ジャネットのアヒルは1日に16個の卵を生みます。ジャネットは毎朝朝食の一環で3個を消費し、毎日4個使って友達向けにマフィンを焼きます。残りを市場で1個あたり2ドルの価格で売ります。彼女は毎日市場でいくら手に入れていますか？」

各カテゴリには、テストデータとChain-of-Thoughtプロンプトのshotに用いる8問の問題と解答が含まれています。

## 必要な依存関係

このスクリプトを実行するには、以下の依存関係が必要です：

```bash
pip install requests google-generativeai
```

また、ローカルでOllamaが実行されている必要があります。Ollamaのインストール方法については、[Ollamaの公式ドキュメント](https://ollama.ai/)を参照してください。

Gemini 2.0 Flashモデルを使用する場合は、Google AI StudioからAPIキーを取得する必要があります。

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
  - `mgsm`: 多言語の算数文章題データセットのみ
  - `all`: すべてのデータセット

- `--api-key`: Gemini 2.0 Flashモデルを使用する場合に必要なAPIキー
  - Google AI StudioからAPIキーを取得する必要があります
  - このオプションは`gemini2.0-flash`モデルを使用する場合のみ必要です

- `--output-dir`: 結果を保存するディレクトリを指定します（デフォルト: `results`）
  - 指定したディレクトリが存在しない場合は自動的に作成されます

- `--save-full-responses`: 完全なモデルの回答を結果に含めるかどうかを指定します
  - このオプションを指定すると、各問題に対するモデルの完全な回答テキストが保存されます
  - ファイルサイズが大きくなる可能性があるため、必要な場合のみ使用してください

例：
```bash
# llama3モデルでzero-shotプロンプトを使用して常識推論のデータセットのみをベンチマーク
python benchmark.py llama3 --shot-type zero_shot --dataset jcommonsenseqa

# mistralモデルでChain-of-Thoughtプロンプトを使用してすべてのデータセットをベンチマーク
python benchmark.py mistral --shot-type shot --dataset all

# Gemini 2.0 Flashモデルを使用してMGSMデータセットをベンチマーク
python benchmark.py gemini2.0-flash --dataset mgsm --shot-type shot --api-key YOUR_API_KEY

# 完全な回答を保存して、カスタム出力ディレクトリを指定
python benchmark.py llama3 --dataset mawps --output-dir custom_results --save-full-responses
```

### 注意事項

- モデル名に含まれるスラッシュやコロンは、アンダースコア（_）に置換されます。
  - 例: `meta/llama3` → `meta_llama3`
- ベンチマーク中は、10問ごとに進捗状況が保存されます。
- APIリクエストの間隔は0.5秒に設定されています。

## 結果の解釈

ベンチマーク結果は `results` ディレクトリ（または `--output-dir` で指定したディレクトリ）に保存されます。各ファイルの命名規則は以下の通りです：

```
<サニタイズされたモデル名>_<データセット名>_<shot_type>_<タイムスタンプ>.json
```

例：
```
llama3_jcommonsenseqa_shot_20250325_171530.json
mistral_mawps_zero_shot_20250325_171530.json
```

結果ファイルには以下の情報が含まれています：

- `dataset`: データセット名
- `shot_type`: プロンプトのタイプ
- `model`: モデル名
- `timestamp`: 実行日時
- `accuracy`: 正解率
- `correct`: 正解数
- `total`: 問題数
- `results`: 各問題の詳細結果
  - `question`: 問題文
  - `answer`: 正解
  - `predicted`: モデルの予測
  - `is_correct`: 正解かどうか
  - `full_response`: モデルの完全な回答テキスト（`--save-full-responses` オプションを指定した場合のみ）

## 進捗状況の確認

ベンチマーク実行中は、10問ごとに進捗状況が保存されます。進捗状況ファイルの命名規則は以下の通りです：

```
<サニタイズされたモデル名>_<データセット名>_<shot_type>_progress_<タイムスタンプ>.json
```

これにより、長時間のベンチマーク実行中に中断した場合でも、途中経過を確認できます。また、タイムスタンプが含まれるため、複数回の実行結果を上書きせずに保存できます。

進捗状況ファイルには、その時点までの正解数、処理済み問題数、正解率、各問題の詳細結果などが含まれています。

## Gemini APIのテスト

このリポジトリには、Gemini APIをテストするための2つのスクリプトが含まれています。

### test_gemini.py

このスクリプトは、Gemini APIの基本的な動作をテストするためのものです。

```bash
python test_gemini.py <API_KEY> [--prompt "プロンプト文"] [--output "出力ファイルパス"]
```

オプション：
- `--prompt`: モデルに送信するプロンプト（デフォルト: "日本の首都はどこですか？"）
- `--output`: 回答結果を保存するファイルパス（デフォルト: "response_YYYYMMDD_HHMMSS.json"）

例：
```bash
# 基本的な使用方法
python test_gemini.py YOUR_API_KEY

# カスタムプロンプトを使用
python test_gemini.py YOUR_API_KEY --prompt "人工知能とは何ですか？"

# 出力ファイルを指定
python test_gemini.py YOUR_API_KEY --output "results/gemini_response.json"
```

出力ファイルには、プロンプト、回答、タイムスタンプが含まれます。

### test_benchmark_gemini.py

このスクリプトは、MGSMデータセットの一部を使用してGemini APIのベンチマークテストを実行します。

```bash
python test_benchmark_gemini.py <API_KEY> [--num-questions 数値] [--output "出力ファイルパス"] [--shot-type shot|zero_shot]
```

オプション：
- `--num-questions`: テストする問題数（デフォルト: 5）
- `--output`: テスト結果を保存するファイルパス（デフォルト: "gemini_mgsm_test_YYYYMMDD_HHMMSS.json"）
- `--shot-type`: プロンプトのタイプ（"shot"または"zero_shot"、デフォルト: "shot"）

例：
```bash
# 基本的な使用方法
python test_benchmark_gemini.py YOUR_API_KEY

# 問題数を指定
python test_benchmark_gemini.py YOUR_API_KEY --num-questions 10

# zero-shotプロンプトを使用
python test_benchmark_gemini.py YOUR_API_KEY --shot-type zero_shot

# 出力ファイルを指定
python test_benchmark_gemini.py YOUR_API_KEY --output "results/gemini_mgsm_test.json"
```

出力ファイルには、テスト結果の概要（正解率、正解数、問題数）と各問題の詳細結果（問題文、正解、予測、正誤判定、完全な回答テキスト）が含まれます。