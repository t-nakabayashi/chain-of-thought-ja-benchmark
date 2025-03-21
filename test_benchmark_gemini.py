import argparse
import json
import os
import re
import time
from pathlib import Path
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


def main():
    parser = argparse.ArgumentParser(description="Test Gemini API with MGSM dataset")
    parser.add_argument("api_key", type=str, help="API key for Gemini models")
    parser.add_argument("--num-questions", type=int, default=5, help="Number of questions to test (default: 5)")

    args = parser.parse_args()

    # MGSMデータセットを読み込む
    dataset_name = "mgsm"
    test_data = load_dataset(f"dataset/{dataset_name}/test.json")
    shot_examples = load_shot_examples(dataset_name, "shot")

    # テスト用に最初のn問だけを使用
    num_questions = min(args.num_questions, len(test_data))
    test_data = test_data[:num_questions]

    results = []
    total = len(test_data)
    correct = 0

    print(f"Testing Gemini API with {total} questions from MGSM dataset")

    for i, item in enumerate(test_data):
        print(f"\nProcessing question {i + 1}/{total}...")
        question = item["question"]
        answer = item["answer"]

        print(f"Question: {question}")
        print(f"Expected answer: {answer}")

        prompt = create_prompt(question, shot_examples, "shot")
        response = query_gemini(prompt, args.api_key)

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

        results.append({"question": question, "answer": answer, "predicted": predicted, "is_correct": is_correct})

        # APIリクエストの間隔を空ける
        time.sleep(1)

    accuracy = correct / total
    print(f"\nFinal Accuracy: {accuracy:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
