import argparse
import json
import os
import re
import time
from pathlib import Path
import requests
import google.generativeai as genai


def sanitize_model_name(model_name):
    """モデル名のスラッシュやコロンを_に置換する"""
    return re.sub(r"[/:]", "_", model_name)


def load_dataset(dataset_path):
    """データセットを読み込む"""
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_shot_examples(dataset_name, shot_type="shot"):
    """shot_exampleまたはzero_shot_exampleを読み込む"""
    shot_path = f"dataset/{dataset_name}/{shot_type}_example.json"
    return load_dataset(shot_path)


def create_prompt(question, shot_examples, shot_type="shot"):
    """プロンプトを作成する"""
    prompt = ""
    for example in shot_examples:
        prompt += example["shot_example"] + "\n"
    prompt += question
    return prompt


def query_ollama(prompt, model_name):
    """ollamaのAPIを使用してモデルに問い合わせる"""
    url = "http://localhost:11434/api/generate"
    data = {"model": model_name, "prompt": prompt, "stream": False}

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        return result["response"]
    except requests.exceptions.RequestException as e:
        print(f"Error querying Ollama: {e}")
        return None


def query_gemini(prompt, api_key, max_retries=5):
    """Gemini APIを使用してモデルに問い合わせる"""
    if not api_key:
        print("Error: API key is required for Gemini models")
        return None

    # Gemini APIの設定
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    retry_count = 0
    while retry_count < max_retries:
        try:
            # モデルに問い合わせ
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                print(f"Error querying Gemini API: {e}")
                print(f"Retrying in 10 seconds... (Attempt {retry_count}/{max_retries})")
                time.sleep(10)
            else:
                print(f"Failed to query Gemini API after {max_retries} attempts: {e}")
                return None


def extract_answer_jcommonsenseqa(response):
    """jcommonsenseqaの回答を抽出する"""
    # デバッグ用に回答全体を表示
    print(f"Debug - Model response: {response[:100]}...")

    # 回答パターン: "答えは(X)XXXです。" または "(X)XXX"
    pattern = r"答えは\s*\(([0-4])\).*です|^\s*\(([0-4])\)"
    match = re.search(pattern, response)
    if match:
        answer_idx = match.group(1) if match.group(1) is not None else match.group(2)
        result = f"({answer_idx})"
        print(f"Debug - Pattern matched: {result}")
        return result

    # 数字だけを探す（0-4の数字）
    pattern2 = r"([0-4])"
    matches = re.findall(pattern2, response)
    if matches:
        # 最後の数字を取得
        result = f"({matches[-1]})"
        print(f"Debug - Number pattern matched: {result}")
        return result

    print("Debug - No pattern matched")
    return None


def is_correct_jcommonsenseqa(predicted, expected):
    """jcommonsenseqaの回答が正解かどうかを判定する"""
    if predicted is None:
        return False

    # 括弧内の数字を抽出
    predicted_pattern = r"\(([0-4])\)"
    expected_pattern = r"\(([0-4])\)"

    predicted_match = re.search(predicted_pattern, predicted)
    expected_match = re.search(expected_pattern, expected)

    if predicted_match and expected_match:
        return predicted_match.group(1) == expected_match.group(1)

    return False


def extract_answer_last_letter_connection(response):
    """last_letter_connectionの回答を抽出する"""
    # デバッグ用に回答全体を表示
    print(f"Debug - Model response: {response[:100]}...")

    # 回答パターン1: "答えは「XXX」です。" または "よって、答えは「XXX」です。"
    pattern1 = r"答えは「([^」]+)」です|よって、答えは「([^」]+)」です"
    match1 = re.search(pattern1, response)
    if match1:
        answer = match1.group(1) if match1.group(1) is not None else match1.group(2)
        print(f"Debug - Pattern1 matched: {answer}")
        return answer

    # 回答パターン2: "答えはXXXです。" または "よって、答えはXXXです。"
    pattern2 = r"答えは([^\s\.。「」]+)です|よって、答えは([^\s\.。「」]+)です"
    match2 = re.search(pattern2, response)
    if match2:
        answer = match2.group(1) if match2.group(1) is not None else match2.group(2)
        print(f"Debug - Pattern2 matched: {answer}")
        return answer

    # 回答パターン3: 「野博」のような2文字の組み合わせを探す
    pattern3 = r"「([一-龠々ぁ-ヶ]{2})」"
    matches = re.findall(pattern3, response)
    if matches:
        answer = matches[-1]
        print(f"Debug - Pattern3 matched: {answer}")
        return answer

    # 回答パターン4: 「野」と「博」を別々に探して結合
    pattern4 = r"「([一-龠々ぁ-ヶ])」.*「([一-龠々ぁ-ヶ])」"
    match4 = re.search(pattern4, response)
    if match4:
        answer = match4.group(1) + match4.group(2)
        print(f"Debug - Pattern4 matched: {answer}")
        return answer

    print("Debug - No pattern matched")
    return None


def is_correct_last_letter_connection(predicted, expected):
    """last_letter_connectionの回答が正解かどうかを判定する"""
    if predicted is None:
        return False

    # 「」を取り除く
    predicted_clean = predicted.replace("「", "").replace("」", "")
    expected_clean = expected.replace("「", "").replace("」", "")

    return predicted_clean == expected_clean


def extract_answer_mawps(response):
    """mawpsの回答を抽出する"""
    # デバッグ用に回答全体を表示
    print(f"Debug - Model response: {response[:100]}...")

    # 回答パターン1: "答えは「XXX」です。" または "よって、答えは「XXX」です。"
    pattern1 = r"答えは「([0-9\.]+)」です|よって、答えは「([0-9\.]+)」です"
    match1 = re.search(pattern1, response)
    if match1:
        answer = match1.group(1) if match1.group(1) is not None else match1.group(2)
        print(f"Debug - Pattern1 matched: {answer}")
        return answer

    # 回答パターン2: "答えはXXXです。" または "よって、答えはXXXです。"
    pattern2 = r"答えは([0-9\.]+)です|よって、答えは([0-9\.]+)です"
    match2 = re.search(pattern2, response)
    if match2:
        answer = match2.group(1) if match2.group(1) is not None else match2.group(2)
        print(f"Debug - Pattern2 matched: {answer}")
        return answer

    # 回答パターン3: 数字だけを探す（最後の数字を取得）
    pattern3 = r"([0-9]+(?:\.[0-9]+)?)"
    matches = re.findall(pattern3, response)
    if matches:
        # 最後の数字を取得
        answer = matches[-1]
        print(f"Debug - Pattern3 matched: {answer}")
        return answer

    print("Debug - No pattern matched")
    return None


def is_correct_mawps(predicted, expected):
    """mawpsの回答が正解かどうかを判定する"""
    if predicted is None:
        return False

    try:
        # 数値に変換して比較
        predicted_num = float(predicted)
        expected_num = float(expected)
        return predicted_num == expected_num
    except (ValueError, TypeError):
        return False


def extract_answer_mgsm(response):
    """mgsmの回答を抽出する"""
    # デバッグ用に回答全体を表示
    print(f"Debug - Model response: {response[:100]}...")

    # 回答パターン1: "答えは「XXX」です。" または "よって、答えは「XXX」です。"
    pattern1 = r"答えは「([0-9\.]+)」です|よって、答えは「([0-9\.]+)」です"
    match1 = re.search(pattern1, response)
    if match1:
        answer = match1.group(1) if match1.group(1) is not None else match1.group(2)
        print(f"Debug - Pattern1 matched: {answer}")
        return answer

    # 回答パターン2: "答えはXXXです。" または "よって、答えはXXXです。"
    pattern2 = r"答えは([0-9\.]+)です|よって、答えは([0-9\.]+)です"
    match2 = re.search(pattern2, response)
    if match2:
        answer = match2.group(1) if match2.group(1) is not None else match2.group(2)
        print(f"Debug - Pattern2 matched: {answer}")
        return answer

    # 回答パターン3: 数字だけを探す（最後の数字を取得）
    pattern3 = r"([0-9]+(?:\.[0-9]+)?)"
    matches = re.findall(pattern3, response)
    if matches:
        # 最後の数字を取得
        answer = matches[-1]
        print(f"Debug - Pattern3 matched: {answer}")
        return answer

    print("Debug - No pattern matched")
    return None


def is_correct_mgsm(predicted, expected):
    """mgsmの回答が正解かどうかを判定する"""
    if predicted is None:
        return False

    try:
        # 数値に変換して比較
        predicted_num = float(predicted)
        expected_num = float(expected)
        return predicted_num == expected_num
    except (ValueError, TypeError):
        return False


def evaluate_jcommonsenseqa(model_name, shot_type="shot", api_key=None, output_dir="results"):
    """jcommonsenseqaのベンチマークを実施する"""
    dataset_name = "jcommonsenseqa"
    test_data = load_dataset(f"dataset/{dataset_name}/test.json")
    shot_examples = load_shot_examples(dataset_name, shot_type)

    results = []
    total = len(test_data)
    correct = 0

    print(f"Evaluating {dataset_name} with {shot_type} prompting...")
    print(f"Total questions: {total}")
    print(f"Model: {model_name}")
    print("=" * 50)

    for i, item in enumerate(test_data):
        print(f"\nProcessing question {i + 1}/{total}...")
        question = item["question"]
        answer = item["answer"]

        print(f"Querying model...")
        prompt = create_prompt(question, shot_examples, shot_type)

        # モデルに応じてクエリ関数を選択
        if model_name == "gemini2.0-flash":
            response = query_gemini(prompt, api_key)
        else:
            response = query_ollama(prompt, model_name)

        if response is None:
            predicted = "Error"
            is_correct = False
            print(f"Error: Failed to get response from model")
        else:
            predicted = extract_answer_jcommonsenseqa(response)
            is_correct = is_correct_jcommonsenseqa(predicted, answer)
            result_str = "✓ Correct" if is_correct else "✗ Incorrect"
            print(f"Result: {result_str}")
            print(f"Expected: {answer}, Predicted: {predicted}")

        if is_correct:
            correct += 1

        # 結果に完全な回答も含める
        result_item = {
            "question": question,
            "answer": answer,
            "predicted": predicted,
            "is_correct": is_correct,
            "full_response": response,
        }
        results.append(result_item)

        if (i + 1) % 10 == 0:
            accuracy_so_far = correct / (i + 1)
            print(f"\n--- Progress: {i + 1}/{total} questions processed ---")
            print(f"Current accuracy: {accuracy_so_far:.4f} ({correct}/{i + 1})")
            print("=" * 50)
            # 進捗状況を保存
            save_progress(dataset_name, shot_type, model_name, results, correct, i + 1, output_dir)

        # APIリクエストの間隔を空ける
        time.sleep(0.5)

    accuracy = correct / total
    print(f"\nFinal Accuracy: {accuracy:.4f} ({correct}/{total})")

    return {
        "dataset": dataset_name,
        "shot_type": shot_type,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


def evaluate_last_letter_connection(model_name, shot_type="shot", api_key=None, output_dir="results"):
    """last_letter_connectionのベンチマークを実施する"""
    dataset_name = "last_letter_connection"
    test_data = load_dataset(f"dataset/{dataset_name}/test.json")
    shot_examples = load_shot_examples(dataset_name, shot_type)

    results = []
    total = len(test_data)
    correct = 0

    print(f"Evaluating {dataset_name} with {shot_type} prompting...")
    print(f"Total questions: {total}")
    print(f"Model: {model_name}")
    print("=" * 50)

    for i, item in enumerate(test_data):
        print(f"\nProcessing question {i + 1}/{total}...")
        question = item["question"]
        answer = item["answer"]

        print(f"Querying model...")
        prompt = create_prompt(question, shot_examples, shot_type)

        # モデルに応じてクエリ関数を選択
        if model_name == "gemini2.0-flash":
            response = query_gemini(prompt, api_key)
        else:
            response = query_ollama(prompt, model_name)

        if response is None:
            predicted = "Error"
            is_correct = False
            print(f"Error: Failed to get response from model")
        else:
            predicted = extract_answer_last_letter_connection(response)
            is_correct = is_correct_last_letter_connection(predicted, answer)
            result_str = "✓ Correct" if is_correct else "✗ Incorrect"
            print(f"Result: {result_str}")
            print(f"Expected: {answer}, Predicted: {predicted}")

        if is_correct:
            correct += 1

        # 結果に完全な回答も含める
        result_item = {
            "question": question,
            "answer": answer,
            "predicted": predicted,
            "is_correct": is_correct,
            "full_response": response,
        }
        results.append(result_item)

        if (i + 1) % 10 == 0:
            accuracy_so_far = correct / (i + 1)
            print(f"\n--- Progress: {i + 1}/{total} questions processed ---")
            print(f"Current accuracy: {accuracy_so_far:.4f} ({correct}/{i + 1})")
            print("=" * 50)
            # 進捗状況を保存
            save_progress(dataset_name, shot_type, model_name, results, correct, i + 1, output_dir)

        # APIリクエストの間隔を空ける
        time.sleep(0.5)

    accuracy = correct / total
    print(f"\nFinal Accuracy: {accuracy:.4f} ({correct}/{total})")

    return {
        "dataset": dataset_name,
        "shot_type": shot_type,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


def evaluate_mawps(model_name, shot_type="shot", api_key=None, output_dir="results"):
    """mawpsのベンチマークを実施する"""
    dataset_name = "mawps"
    test_data = load_dataset(f"dataset/{dataset_name}/test.json")
    shot_examples = load_shot_examples(dataset_name, shot_type)

    results = []
    total = len(test_data)
    correct = 0

    print(f"Evaluating {dataset_name} with {shot_type} prompting...")
    print(f"Total questions: {total}")
    print(f"Model: {model_name}")
    print("=" * 50)

    for i, item in enumerate(test_data):
        print(f"\nProcessing question {i + 1}/{total}...")
        question = item["question"]
        answer = item["answer"]

        print(f"Querying model...")
        prompt = create_prompt(question, shot_examples, shot_type)

        # モデルに応じてクエリ関数を選択
        if model_name == "gemini2.0-flash":
            response = query_gemini(prompt, api_key)
        else:
            response = query_ollama(prompt, model_name)

        if response is None:
            predicted = "Error"
            is_correct = False
            print(f"Error: Failed to get response from model")
        else:
            predicted = extract_answer_mawps(response)
            is_correct = is_correct_mawps(predicted, answer)
            result_str = "✓ Correct" if is_correct else "✗ Incorrect"
            print(f"Result: {result_str}")
            print(f"Expected: {answer}, Predicted: {predicted}")

        if is_correct:
            correct += 1

        # 結果に完全な回答も含める
        result_item = {
            "question": question,
            "answer": answer,
            "predicted": predicted,
            "is_correct": is_correct,
            "full_response": response,
        }
        results.append(result_item)

        if (i + 1) % 10 == 0:
            accuracy_so_far = correct / (i + 1)
            print(f"\n--- Progress: {i + 1}/{total} questions processed ---")
            print(f"Current accuracy: {accuracy_so_far:.4f} ({correct}/{i + 1})")
            print("=" * 50)
            # 進捗状況を保存
            save_progress(dataset_name, shot_type, model_name, results, correct, i + 1, output_dir)

        # APIリクエストの間隔を空ける
        time.sleep(0.5)

    accuracy = correct / total
    print(f"\nFinal Accuracy: {accuracy:.4f} ({correct}/{total})")

    return {
        "dataset": dataset_name,
        "shot_type": shot_type,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


def evaluate_mgsm(model_name, shot_type="shot", api_key=None, output_dir="results"):
    """mgsmのベンチマークを実施する"""
    dataset_name = "mgsm"
    test_data = load_dataset(f"dataset/{dataset_name}/test.json")
    shot_examples = load_shot_examples(dataset_name, shot_type)

    results = []
    total = len(test_data)
    correct = 0

    print(f"Evaluating {dataset_name} with {shot_type} prompting...")
    print(f"Total questions: {total}")
    print(f"Model: {model_name}")
    print("=" * 50)

    for i, item in enumerate(test_data):
        print(f"\nProcessing question {i + 1}/{total}...")
        question = item["question"]
        answer = item["answer"]

        print(f"Querying model...")
        prompt = create_prompt(question, shot_examples, shot_type)

        # モデルに応じてクエリ関数を選択
        if model_name == "gemini2.0-flash":
            response = query_gemini(prompt, api_key)
        else:
            response = query_ollama(prompt, model_name)

        if response is None:
            predicted = "Error"
            is_correct = False
            print(f"Error: Failed to get response from model")
        else:
            predicted = extract_answer_mgsm(response)
            is_correct = is_correct_mgsm(predicted, answer)
            result_str = "✓ Correct" if is_correct else "✗ Incorrect"
            print(f"Result: {result_str}")
            print(f"Expected: {answer}, Predicted: {predicted}")

        if is_correct:
            correct += 1

        # 結果に完全な回答も含める
        result_item = {
            "question": question,
            "answer": answer,
            "predicted": predicted,
            "is_correct": is_correct,
            "full_response": response,
        }
        results.append(result_item)

        if (i + 1) % 10 == 0:
            accuracy_so_far = correct / (i + 1)
            print(f"\n--- Progress: {i + 1}/{total} questions processed ---")
            print(f"Current accuracy: {accuracy_so_far:.4f} ({correct}/{i + 1})")
            print("=" * 50)
            # 進捗状況を保存
            save_progress(dataset_name, shot_type, model_name, results, correct, i + 1, output_dir)

        # APIリクエストの間隔を空ける
        time.sleep(0.5)

    accuracy = correct / total
    print(f"\nFinal Accuracy: {accuracy:.4f} ({correct}/{total})")

    return {
        "dataset": dataset_name,
        "shot_type": shot_type,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


def save_progress(dataset_name, shot_type, model_name, results, correct, processed, output_dir=None):
    """進捗状況を保存する"""
    sanitized_model_name = sanitize_model_name(model_name)

    # 出力ディレクトリが指定されていない場合は、デフォルトのresultsディレクトリを使用
    if output_dir is None:
        results_dir = Path("results")
    else:
        results_dir = Path(output_dir)

    results_dir.mkdir(exist_ok=True, parents=True)

    # タイムスタンプを含むファイル名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    progress_file = results_dir / f"{sanitized_model_name}_{dataset_name}_{shot_type}_progress_{timestamp}.json"

    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "shot_type": shot_type,
                "model": model_name,
                "processed": processed,
                "correct": correct,
                "accuracy_so_far": correct / processed,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Progress saved to {progress_file}")
    return progress_file


def save_results(results, model_name, dataset_name, shot_type, output_file=None):
    """結果を保存する"""
    sanitized_model_name = sanitize_model_name(model_name)

    # 出力ファイルが指定されていない場合は、デフォルトのファイル名を生成
    if output_file is None:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True, parents=True)

        # タイムスタンプを含むファイル名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"{sanitized_model_name}_{dataset_name}_{shot_type}_{timestamp}.json"
    else:
        results_file = Path(output_file)
        results_file.parent.mkdir(exist_ok=True, parents=True)

    # 結果に追加情報を付加
    enhanced_results = {
        "dataset": dataset_name,
        "shot_type": shot_type,
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(enhanced_results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {results_file}")
    return results_file


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLMs on Japanese Chain-of-Thought datasets")
    parser.add_argument("model_name", type=str, help="Name of the model to benchmark")
    parser.add_argument(
        "--shot-type", type=str, choices=["shot", "zero_shot"], default="shot", help="Type of prompting to use (default: shot)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["jcommonsenseqa", "last_letter_connection", "mawps", "mgsm", "all"],
        default="all",
        help="Dataset to benchmark (default: all)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for Gemini models (required when model_name is 'gemini2.0-flash')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results)",
    )
    parser.add_argument(
        "--save-full-responses",
        action="store_true",
        help="Save full model responses in the results (may increase file size)",
    )

    args = parser.parse_args()

    # Geminiモデルの場合はAPIキーが必要
    if args.model_name == "gemini2.0-flash" and not args.api_key:
        print("Error: API key is required for Gemini models")
        print("Please provide an API key using the --api-key option")
        return

    # モデル名のサニタイズ
    sanitized_model_name = sanitize_model_name(args.model_name)
    print(f"Benchmarking model: {args.model_name} (sanitized: {sanitized_model_name})")
    print(f"Shot type: {args.shot_type}")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output_dir}")
    print(f"Save full responses: {args.save_full_responses}")

    # 結果ディレクトリの作成
    results_dir = Path(args.output_dir)
    results_dir.mkdir(exist_ok=True, parents=True)

    # タイムスタンプ（すべてのデータセットで共通）
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if args.dataset == "jcommonsenseqa" or args.dataset == "all":
        results = evaluate_jcommonsenseqa(args.model_name, args.shot_type, args.api_key, args.output_dir)
        # 完全な回答を保存しない場合は、resultsから削除
        if not args.save_full_responses:
            for item in results["results"]:
                if "full_response" in item:
                    del item["full_response"]
        output_file = results_dir / f"{sanitized_model_name}_jcommonsenseqa_{args.shot_type}_{timestamp}.json"
        save_results(results, args.model_name, "jcommonsenseqa", args.shot_type, output_file)

    if args.dataset == "last_letter_connection" or args.dataset == "all":
        results = evaluate_last_letter_connection(args.model_name, args.shot_type, args.api_key, args.output_dir)
        # 完全な回答を保存しない場合は、resultsから削除
        if not args.save_full_responses:
            for item in results["results"]:
                if "full_response" in item:
                    del item["full_response"]
        output_file = results_dir / f"{sanitized_model_name}_last_letter_connection_{args.shot_type}_{timestamp}.json"
        save_results(results, args.model_name, "last_letter_connection", args.shot_type, output_file)

    if args.dataset == "mawps" or args.dataset == "all":
        results = evaluate_mawps(args.model_name, args.shot_type, args.api_key, args.output_dir)
        # 完全な回答を保存しない場合は、resultsから削除
        if not args.save_full_responses:
            for item in results["results"]:
                if "full_response" in item:
                    del item["full_response"]
        output_file = results_dir / f"{sanitized_model_name}_mawps_{args.shot_type}_{timestamp}.json"
        save_results(results, args.model_name, "mawps", args.shot_type, output_file)

    if args.dataset == "mgsm" or args.dataset == "all":
        results = evaluate_mgsm(args.model_name, args.shot_type, args.api_key, args.output_dir)
        # 完全な回答を保存しない場合は、resultsから削除
        if not args.save_full_responses:
            for item in results["results"]:
                if "full_response" in item:
                    del item["full_response"]
        output_file = results_dir / f"{sanitized_model_name}_mgsm_{args.shot_type}_{timestamp}.json"
        save_results(results, args.model_name, "mgsm", args.shot_type, output_file)


if __name__ == "__main__":
    main()
