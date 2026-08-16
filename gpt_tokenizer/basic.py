"""
basic.py
--------
BasicTokenizer: the simplest possible byte-level BPE tokenizer. It treats
the entire input as ONE sequence of bytes -- no regex pre-splitting, no
special tokens. This is the version to read first to understand the core
BPE algorithm; RegexTokenizer adds the production-grade pieces on top.
"""

from __future__ import annotations

from .base import Tokenizer
from .helpers import get_stats, merge


class BasicTokenizer(Tokenizer):

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= 256, "vocab_size must be >= 256"
        num_merges = vocab_size - 256

        ids = list(text.encode("utf-8"))
        merges: dict[tuple[int, int], int] = {}
        vocab = self._base_vocab()

        for i in range(num_merges):
            stats = get_stats([ids])
            if not stats:
                break
            pair = max(stats, key=stats.get)
            new_id = 256 + i
            ids = merge(ids, pair, new_id)
            merges[pair] = new_id
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {new_id} "
                      f"({vocab[new_id]!r}) had {stats[pair]} occurrences")

        self.merges = merges
        self._rebuild_vocab()

    def encode(self, text: str, **kwargs) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats([ids])
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids
