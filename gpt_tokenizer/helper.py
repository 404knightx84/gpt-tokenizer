"""
helpers.py
----------
Pure, stateless functions used by the BPE algorithm. Kept separate from the
Tokenizer classes so they're trivially unit-testable and reusable.
"""

from __future__ import annotations


def get_stats(ids_list: list[list[int]], counts: dict | None = None) -> dict:
    """Count adjacent-pair frequencies across one or more id sequences.

    ids_list: a list of token-id sequences (e.g. one per pre-split chunk).
    counts:   an existing dict to accumulate into (used for incremental
              updates); if None, a fresh dict is created.
    """
    counts = {} if counts is None else counts
    for ids in ids_list:
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    new_ids = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def render_token(token_bytes: bytes) -> str:
    """Pretty-print a token's bytes for debugging/inspection, replacing
    unprintable characters so terminal output doesn't break."""
    try:
        s = token_bytes.decode("utf-8")
    except UnicodeDecodeError:
        s = str(token_bytes)
    return s
