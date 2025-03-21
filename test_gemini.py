import argparse
import json
import os
import re
import time
from pathlib import Path
import google.generativeai as genai


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


def main():
    parser = argparse.ArgumentParser(description="Test Gemini API")
    parser.add_argument("api_key", type=str, help="API key for Gemini models")

    args = parser.parse_args()

    # テスト用のプロンプト
    prompt = "日本の首都はどこですか？"

    print(f"Testing Gemini API with prompt: {prompt}")
    response = query_gemini(prompt, args.api_key)

    if response:
        print(f"Response: {response}")
        print("Test successful!")
    else:
        print("Test failed!")


if __name__ == "__main__":
    main()
