# 設計仕様書：日本語Chain-of-Thoughtデータセットベンチマーク

## 1. プロジェクトの目的

このプロジェクトの目的は、[nlp-waseda/chain-of-thought-ja-dataset](https://github.com/nlp-waseda/chain-of-thought-ja-dataset)の日本語Chain-of-Thoughtデータセットを使用して、ローカルで実行されている言語モデル（LLM）の日本語での推論能力を評価することです。特に、Chain-of-Thought（思考の連鎖）プロンプトを使用した場合と使用しない場合の性能差を測定することを目的としています。

## 2. システム構成

システムは以下のコンポーネントで構成されています：

1. **ベンチマークスクリプト（benchmark.py）**：メインのスクリプトで、データセットの読み込み、モデルへの問い合わせ、結果の評価と保存を行います。
2. **データセット**：4つのカテゴリ（jcommonsenseqa、last_letter_connection、mawps、mgsm）のデータセットが含まれています。
3. **Ollamaサーバー**：ローカルで実行されるLLMサーバーで、APIを通じてモデルに問い合わせを行います。
4. **Google Gemini API**：Google Gemini 2.0 Flashモデルに問い合わせるためのAPIです。

## 3. モジュール設計

ベンチマークスクリプト（benchmark.py）は、以下の主要な関数で構成されています：

### 3.1 ユーティリティ関数

- `sanitize_model_name(model_name)`: モデル名のスラッシュやコロンをアンダースコアに置換する関数
- `load_dataset(dataset_path)`: データセットを読み込む関数
- `load_shot_examples(dataset_name, shot_type)`: shot_exampleまたはzero_shot_exampleを読み込む関数
- `create_prompt(question, shot_examples, shot_type)`: プロンプトを作成する関数
- `query_ollama(prompt, model_name)`: ollamaのAPIを使用してモデルに問い合わせる関数
- `query_gemini(prompt, api_key, max_retries=5)`: Google Gemini APIを使用してモデルに問い合わせる関数

### 3.2 回答抽出関数

- `extract_answer_jcommonsenseqa(response)`: jcommonsenseqaの回答を抽出する関数
- `extract_answer_last_letter_connection(response)`: last_letter_connectionの回答を抽出する関数
- `extract_answer_mawps(response)`: mawpsの回答を抽出する関数
- `extract_answer_mgsm(response)`: mgsmの回答を抽出する関数

### 3.3 回答評価関数

- `is_correct_jcommonsenseqa(predicted, expected)`: jcommonsenseqaの回答が正解かどうかを判定する関数
- `is_correct_last_letter_connection(predicted, expected)`: last_letter_connectionの回答が正解かどうかを判定する関数
- `is_correct_mawps(predicted, expected)`: mawpsの回答が正解かどうかを判定する関数
- `is_correct_mgsm(predicted, expected)`: mgsmの回答が正解かどうかを判定する関数

### 3.4 ベンチマーク実行関数

- `evaluate_jcommonsenseqa(model_name, shot_type, api_key=None)`: jcommonsenseqaのベンチマークを実施する関数
- `evaluate_last_letter_connection(model_name, shot_type, api_key=None)`: last_letter_connectionのベンチマークを実施する関数
- `evaluate_mawps(model_name, shot_type, api_key=None)`: mawpsのベンチマークを実施する関数
- `evaluate_mgsm(model_name, shot_type, api_key=None)`: mgsmのベンチマークを実施する関数

### 3.5 結果保存関数

- `save_progress(dataset_name, shot_type, model_name, results, correct, processed)`: 進捗状況を保存する関数
- `save_results(results, model_name, dataset_name, shot_type)`: 結果を保存する関数

### 3.6 メイン関数

- `main()`: コマンドライン引数を解析し、指定されたデータセットとモデルでベンチマークを実行する関数

## 4. データフロー

1. ユーザーがコマンドラインからベンチマークスクリプトを実行し、モデル名、データセット、プロンプトタイプを指定します。Gemini 2.0 Flashモデルを使用する場合は、APIキーも指定します。
2. スクリプトは指定されたデータセットとshot_exampleを読み込みます。
3. 各問題に対して、プロンプトを作成します。
   - Ollamaモデルの場合は、Ollamaサーバーを通じてモデルに問い合わせます。
   - Gemini 2.0 Flashモデルの場合は、Google Gemini APIを通じてモデルに問い合わせます。
4. モデルの回答から答えを抽出し、正解と比較して正誤を判定します。
5. 10問ごとに進捗状況を保存し、最終的な結果をJSONファイルとして保存します。

## 5. 実装の詳細

### 5.1 回答抽出の実装

各データセットの回答抽出は、正規表現を使用して実装されています。モデルの回答から答えを抽出するために、複数のパターンを試行します。

#### jcommonsenseqa

1. 「答えは(X)XXXです。」または「(X)XXX」のパターンを探します。
2. 数字（0-4）を探します。

#### last_letter_connection

1. 「答えは「XXX」です。」または「よって、答えは「XXX」です。」のパターンを探します。
2. 「答えはXXXです。」または「よって、答えはXXXです。」のパターンを探します。
3. 「野」と「博」のような2文字の組み合わせを探します。
4. 「野」と「博」を別々に探して結合します。

#### mawps

1. 「答えは「XXX」です。」または「よって、答えは「XXX」です。」のパターンを探します。
2. 「答えはXXXです。」または「よって、答えはXXXです。」のパターンを探します。
3. 数字を探します。

#### mgsm

1. 「答えは「XXX」です。」または「よって、答えは「XXX」です。」のパターンを探します。
2. 「答えはXXXです。」または「よって、答えはXXXです。」のパターンを探します。
3. 数字を探します。

### 5.2 回答評価の実装

各データセットの回答評価は、以下のように実装されています：

#### jcommonsenseqa

括弧内の数字が一致していれば正解と判定します。

#### last_letter_connection

「」（かぎかっこ）を取り除いた文字列が一致していれば正解と判定します。

#### mawps

数値に変換して比較し、一致していれば正解と判定します。

#### mgsm

数値に変換して比較し、一致していれば正解と判定します。

### 5.3 デバッグ機能

各回答抽出関数には、デバッグ情報を表示する機能が実装されています。これにより、モデルの回答と抽出された答えを確認することができます。

## 6. 拡張性と将来の改善点

### 6.1 他のLLMフレームワークへの対応

現在はOllamaとGoogle Gemini APIに対応していますが、将来的には以下のフレームワークにも対応することが考えられます：

- llama.cpp
- LocalAI
- Text Generation WebUI
- OpenAI API
- Anthropic API
- その他のLLMフレームワーク

### 6.2 回答抽出の改善

現在の回答抽出は正規表現を使用していますが、より高度な自然言語処理技術を使用することで、抽出精度を向上させることができます。

### 6.3 ベンチマーク結果の可視化

現在はJSONファイルとして結果を保存していますが、将来的にはグラフやチャートを使用して結果を可視化する機能を追加することが考えられます。

### 6.4 並列処理の実装

現在は1つの問題ずつ順番に処理していますが、並列処理を実装することで、ベンチマークの実行時間を短縮することができます。

### 6.5 中断からの再開機能

現在は10問ごとに進捗状況を保存していますが、中断した場合に途中から再開する機能を追加することが考えられます。

## 7. 結論

このプロジェクトは、日本語Chain-of-Thoughtデータセットを使用して、LLMの日本語での推論能力を評価するためのツールを提供します。現在はOllamaとGoogle Gemini APIに対応しており、ローカルモデルとクラウドモデルの両方を評価することができます。将来的には他のフレームワークにも対応することで、より幅広いモデルの評価が可能になります。

### 7.1 Gemini APIの実装

Gemini APIの実装では、以下の機能を追加しました：

1. APIキーを使用してGemini 2.0 Flashモデルに問い合わせる機能
2. エラー処理（APIキーがない場合のエラーメッセージ、リトライ処理）
3. コマンドライン引数でAPIキーを指定する機能

これにより、ローカルモデルだけでなく、クラウドベースの最新モデルでもベンチマークを実行できるようになりました。