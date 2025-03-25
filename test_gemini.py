import argparse
import json
import os
import re
import time
from pathlib import Path
import google.generativeai as genai
from datetime import datetime


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


def save_response(prompt, response, output_file=None):
    """回答結果をファイルに保存する"""
    # 出力ファイルが指定されていない場合は、現在の日時を使用してファイル名を生成
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"response_{timestamp}.json"

    # 結果ディレクトリの作成
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    # 結果を辞書形式で保存
    result = {"prompt": prompt, "response": response, "timestamp": datetime.now().isoformat()}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Response saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Test Gemini API")
    parser.add_argument("api_key", type=str, help="API key for Gemini models")
    parser.add_argument("--prompt", type=str, default="日本の首都はどこですか？", help="Prompt to send to the model")
    parser.add_argument(
        "--output", type=str, help="Output file path to save the response (default: response_YYYYMMDD_HHMMSS.json)"
    )

    args = parser.parse_args()

    # プロンプト
    prompt = args.prompt

    print(f"Testing Gemini API with prompt: {prompt}")
    response = query_gemini(prompt, args.api_key)

    if response:
        print(f"Response: {response}")
        # 回答結果をファイルに保存
        output_file = save_response(prompt, response, args.output)
        print("Test successful!")
    else:
        print("Test failed!")


if __name__ == "__main__":
    main()
