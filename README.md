# gpt-tokenizer

A from-scratch, byte-level Byte Pair Encoding (BPE) tokenizer, built the
same way GPT-2 / GPT-4 tokenizers actually work — no external tokenizer
libraries (`tiktoken`, `transformers`, etc.), just Python + the `regex`
package for unicode-aware pattern matching.

## File structure

```
gpt_tokenizer_project/
├── gpt_tokenizer/                 # the library (importable package)
│   ├── __init__.py                # public API: BasicTokenizer, RegexTokenizer
│   ├── helpers.py                 # pure functions: get_stats(), merge(), render_token()
│   ├── patterns.py                # GPT2_SPLIT_PATTERN, GPT4_SPLIT_PATTERN
│   ├── base.py                    # Tokenizer base class: vocab, special tokens, save/load, decode
│   ├── basic.py                   # BasicTokenizer(Tokenizer)  — pure BPE, no pre-splitting
│   └── regex_tokenizer.py         # RegexTokenizer(Tokenizer)  — + regex chunking + special tokens
├── tests/
│   └── test_tokenizer.py          # round-trip, chunk-boundary, special-token, save/load tests
├── scripts/
│   ├── train.py                   # CLI: train a tokenizer on a text file, save the model
│   └── encode_decode.py           # CLI: load a model, encode/decode a string
├── data/
│   └── sample.txt                 # tiny sample corpus to train on
├── models/                        # (created at train time) saved .model / .vocab files
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Design architecture

### Why split it this way

| Concern | Where it lives | Why separated |
|---|---|---|
| Pair-counting / merging math | `helpers.py` | Pure functions, no state — trivial to unit test in isolation, reused by every tokenizer variant. |
| Pre-tokenization patterns | `patterns.py` | The regex is a *policy choice* (GPT-2 vs GPT-4 style), not core algorithm — kept swappable. |
| Vocab / special tokens / persistence | `base.py` | Identical across every BPE variant regardless of how training/encoding works — one implementation, no duplication. |
| Training + encoding strategy | `basic.py`, `regex_tokenizer.py` | The only parts that actually differ between "toy" and "production" tokenizers — isolated so you can read `basic.py` first to learn the algorithm, then see exactly what `regex_tokenizer.py` adds. |
| CLI / usage | `scripts/` | Keeps the library importable and dependency-free of `argparse` plumbing; scripts are thin wrappers. |

### Class hierarchy

```
Tokenizer (base.py)
│   - self.merges: dict[(id1,id2) -> new_id]      learned merge rules, in order
│   - self.vocab:  dict[id -> bytes]               id -> raw bytes it expands to
│   - self.special_tokens: dict[str -> id]         reserved ids that skip BPE
│   + register_special_tokens()
│   + decode()            (shared: ids -> bytes -> str)
│   + save() / load()     (shared: portable .model text format)
│   - train()  [abstract]
│   - encode() [abstract]
│
├── BasicTokenizer          whole text = one byte sequence, no regex, no specials
│   + train()   plain BPE loop over the full byte sequence
│   + encode()  apply learned merges in learned order
│
└── RegexTokenizer          production-style
    + __init__(pattern)     compiles a pre-split pattern (patterns.py)
    + train()   split into chunks first, then run BPE *within* each chunk
    + encode_ordinary()     same idea at inference time
    + encode()   handles special tokens (e.g. <|endoftext|>) by splitting
                 them out of the text before ordinary encoding runs
```

### Data flow

**Training** (`RegexTokenizer.train`):
```
raw text
  -> regex pre-split -> [chunk1, chunk2, ...]
  -> utf-8 encode each chunk -> [[bytes], [bytes], ...]
  -> repeat until vocab_size reached:
       count all adjacent pairs across all chunks (helpers.get_stats)
       pick the most frequent pair
       replace every occurrence with a new token id (helpers.merge)
       record the rule in self.merges, extend self.vocab
```

**Encoding** (`RegexTokenizer.encode`):
```
text
  -> split out any allowed special tokens (regex alternation) -> parts
  -> for each non-special part:
       regex pre-split -> chunks
       utf-8 encode each chunk -> byte ids
       repeatedly apply the EARLIEST-LEARNED applicable merge
         (earliest-learned = correct BPE priority order)
     until no merge in self.merges applies
  -> concatenate all ids (special-token ids inserted verbatim)
```

**Decoding** (`Tokenizer.decode`, shared by both variants):
```
ids -> look each id up in self.vocab -> bytes -> b"".join(...) -> utf-8 decode
```

### Key design decisions worth knowing

- **Byte-level base vocab (ids 0–255)**: guarantees any input — any
  language, emoji, or malformed text — can always be encoded. There is no
  "unknown token" fallback needed.
- **Regex pre-splitting before BPE**: prevents a whitespace/punctuation
  token from ever fusing into a word token, which keeps the vocabulary
  from being polluted by contextual accidents (e.g. `"dog."` vs `"dog"`).
- **`regex` package, not stdlib `re`**: `\p{L}` / `\p{N}` unicode property
  classes aren't supported by stdlib `re`.
- **Merges stored in learned order, in a dict**: encoding replays them in
  that exact order (lowest new-id first) so tokenization at inference
  time exactly matches how the vocab was built during training.

## Quick start

```bash
pip install -r requirements.txt

# train a tokenizer on the sample corpus
python scripts/train.py --input data/sample.txt --vocab-size 512 --out models/mytok

# encode/decode with it
python scripts/encode_decode.py --model models/mytok.model --text "The quick brown fox 🚀"

# run the tests
python tests/test_tokenizer.py
```

```python
# ...or use it as a library directly
from gpt_tokenizer import RegexTokenizer

tok = RegexTokenizer()
tok.train(open("data/sample.txt").read(), vocab_size=512)
tok.register_special_tokens({"<|endoftext|>": 512})

ids = tok.encode("hello world<|endoftext|>", allowed_special="all")
assert tok.decode(ids) == "hello world<|endoftext|>"
```

## What's intentionally left out (next steps if you want them)

- **Training speed**: the current loop rescans all pairs every merge
  (O(n) per merge). Real tokenizers use a max-heap + linked-list so only
  the pairs affected by the last merge are re-counted.
- **Parallel training**: chunk-level counting is embarrassingly parallel
  across cores/processes; not implemented here for clarity.
- **A `tiktoken`-compatible binary vocab format**: this repo uses a
  simple, readable text `.model` format instead.
