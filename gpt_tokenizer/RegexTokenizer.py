"""
regex_tokenizer.py
-------------------
RegexTokenizer: the GPT-2 / GPT-4 style tokenizer. Adds two things on top
of BasicTokenizer:
  1. Regex pre-splitting (patterns.py) so BPE merges never cross chunk
     boundaries (word / number / punctuation / whitespace stay separate).
  2. Special-token handling (e.g. <|endoftext|>) that bypasses BPE entirely
     and maps straight to a reserved id.
"""

from __future__ import annotations

import regex as re  # third-party `regex`, not stdlib `re` -- needed for \p{L}/\p{N}

from .base import Tokenizer
from .helpers import get_stats, merge
from .patterns import GPT4_SPLIT_PATTERN


class RegexTokenizer(Tokenizer):

    def __init__(self, pattern: str = GPT4_SPLIT_PATTERN):
        super().__init__()
        self._compile_pattern(pattern)

    def _compile_pattern(self, pattern: str) -> None:
        self.pattern = pattern
        self._compiled = re.compile(pattern)

    # -- training ------------------------------------------------------------
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= 256, "vocab_size must be >= 256"
        num_merges = vocab_size - 256

        # split into chunks first; each chunk's ids are merged independently
        text_chunks = self._compiled.findall(text)
        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        merges: dict[tuple[int, int], int] = {}
        vocab = self._base_vocab()

        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))  # deterministic tie-break
            new_id = 256 + i
            ids = [merge(chunk_ids, pair, new_id) for chunk_ids in ids]
            merges[pair] = new_id
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {new_id} "
                      f"({vocab[new_id]!r}) had {stats[pair]} occurrences")

        self.merges = merges
        self._rebuild_vocab()

    # -- encoding ----------------------------------------------------------
    def _encode_chunk(self, chunk_bytes: bytes) -> list[int]:
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            stats = get_stats([ids])
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text known to contain no special tokens."""
        ids = []
        for chunk in self._compiled.findall(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def encode(self, text: str, allowed_special: str | set = "none_raise") -> list[int]:
        special = self.special_tokens
        if not special:
            return self.encode_ordinary(text)

        if allowed_special == "all":
            allowed = set(special)
        elif allowed_special == "none":
            allowed = set()
        elif allowed_special == "none_raise":
            allowed = set()
            for token_str in special:
                assert token_str not in text, f"special token {token_str!r} found in text"
        elif isinstance(allowed_special, set):
            allowed = allowed_special
        else:
            raise ValueError(f"unknown allowed_special setting: {allowed_special!r}")

        if not allowed:
            return self.encode_ordinary(text)

        split_pattern = "(" + "|".join(re.escape(t) for t in allowed) + ")"
        parts = re.split(split_pattern, text)
        ids = []
        for part in parts:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids
