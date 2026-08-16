from __future__ import annotations

MODEL_VERSION = "gpt-tokenizer v1"


class Tokenizer:
    """Common interface + shared state. Subclasses must implement
    `train()` and `encode()`."""

    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}   # (id1, id2) -> new_id, in learned order
        self.special_tokens: dict[str, int] = {}         # e.g. {"<|endoftext|>": 100257}
        self.vocab: dict[int, bytes] = self._base_vocab()

    # -- vocab construction --------------------------------------------------
    @staticmethod
    def _base_vocab() -> dict[int, bytes]:
        return {idx: bytes([idx]) for idx in range(256)}

    def _rebuild_vocab(self) -> None:
        """Recompute self.vocab from scratch: 256 byte tokens, then merges
        applied in the order they were learned, then special tokens."""
        vocab = self._base_vocab()
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for token_str, idx in self.special_tokens.items():
            vocab[idx] = token_str.encode("utf-8")
        self.vocab = vocab

    def register_special_tokens(self, special_tokens: dict[str, int]) -> None:
        """special_tokens: e.g. {"<|endoftext|>": 256 + num_merges}"""
        self.special_tokens = dict(special_tokens)
        self._rebuild_vocab()

    # -- interface subclasses must implement ----------------------------------
    def train(self, text: str, vocab_size: int, verbose: bool = False):
        raise NotImplementedError

    def encode(self, text: str, **kwargs) -> list[int]:
        raise NotImplementedError

    # -- shared decode ---------------------------------------------------------
    def decode(self, ids: list[int]) -> str:
        parts = []
        for idx in ids:
            if idx not in self.vocab:
                raise ValueError(f"unknown token id: {idx}")
            parts.append(self.vocab[idx])
        return b"".join(parts).decode("utf-8", errors="replace")

    # -- persistence -------------------------------------------------------
    def save(self, file_prefix: str) -> None:
        """Writes `{file_prefix}.model` (merges + special tokens + pattern)
        and `{file_prefix}.vocab` (a human-readable inspection file)."""
        model_path = f"{file_prefix}.model"
        with open(model_path, "w", encoding="utf-8") as f:
            f.write(f"{MODEL_VERSION}\n")
            f.write(f"{getattr(self, 'pattern', '')}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for token_str, idx in self.special_tokens.items():
                f.write(f"{token_str} {idx}\n")
            for (p0, p1), idx in self.merges.items():
                f.write(f"{p0} {p1} {idx}\n")

        vocab_path = f"{file_prefix}.vocab"
        from .helpers import render_token
        with open(vocab_path, "w", encoding="utf-8") as f:
            for idx, token_bytes in sorted(self.vocab.items()):
                f.write(f"[{idx}] {render_token(token_bytes)!r}\n")

    def load(self, model_file: str) -> None:
        merges: dict[tuple[int, int], int] = {}
        special_tokens: dict[str, int] = {}
        with open(model_file, "r", encoding="utf-8") as f:
            version = f.readline().strip()
            assert version == MODEL_VERSION, f"unsupported model file: {version!r}"
            pattern = f.readline().strip()
            if pattern and hasattr(self, "_compile_pattern"):
                self._compile_pattern(pattern)
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                token_str, idx = f.readline().rsplit(" ", 1)
                special_tokens[token_str] = int(idx)
            for line in f:
                p0, p1, idx = map(int, line.split())
                merges[(p0, p1)] = idx
        self.merges = merges
        self.special_tokens = special_tokens
        self._rebuild_vocab()
