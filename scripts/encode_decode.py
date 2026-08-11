import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpt_tokenizer import RegexTokenizer


def main():
    parser = argparse.ArgumentParser(description="Encode/decode text with a trained tokenizer.")
    parser.add_argument("--model", required=True, help="path to a .model file")
    parser.add_argument("--text", required=True, help="text to encode")
    args = parser.parse_args()

    tok = RegexTokenizer()
    tok.load(args.model)

    ids = tok.encode(args.text, allowed_special="all")
    decoded = tok.decode(ids)

    print("text     :", args.text)
    print("token ids:", ids)
    print("num tokens:", len(ids))
    print("decoded  :", decoded)
    print("round-trip OK:", decoded == args.text)


if __name__ == "__main__":
    main()
