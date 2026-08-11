import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpt_tokenizer import BasicTokenizer, RegexTokenizer


def main():
    parser = argparse.ArgumentParser(description="Train a from-scratch BPE tokenizer.")
    parser.add_argument("--input", required=True, help="path to a UTF-8 text file")
    parser.add_argument("--vocab-size", type=int, default=512, help="final vocab size (>=256)")
    parser.add_argument("--out", default="models/tokenizer", help="output file prefix")
    parser.add_argument("--kind", choices=["basic", "regex"], default="regex")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    tok = BasicTokenizer() if args.kind == "basic" else RegexTokenizer()

    t0 = time.time()
    tok.train(text, vocab_size=args.vocab_size, verbose=args.verbose)
    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tok.save(args.out)

    print(f"\nTrained {args.kind} tokenizer: {len(tok.vocab)} tokens "
          f"({len(tok.merges)} merges) in {elapsed:.2f}s")
    print(f"Saved -> {args.out}.model  (and {args.out}.vocab for inspection)")


if __name__ == "__main__":
    main()
