
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpt_tokenizer import BasicTokenizer, RegexTokenizer

SAMPLE = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox jumps again! 123 123 123. "
    "¡Tokenización rápida! 🚀"
)


def test_basic_roundtrip():
    tok = BasicTokenizer()
    tok.train(SAMPLE, vocab_size=300)
    for text in [SAMPLE, "hello world", "", "🚀🚀🚀", "a"]:
        assert tok.decode(tok.encode(text)) == text


def test_regex_roundtrip():
    tok = RegexTokenizer()
    tok.train(SAMPLE, vocab_size=300)
    for text in [SAMPLE, "hello, world!", "", "999999999", "  spaced  out  "]:
        assert tok.decode(tok.encode(text)) == text


def test_regex_chunks_dont_bridge_punctuation():
    tok = RegexTokenizer()
    tok.train("dog. dog. dog. dog. cat! cat! cat! cat!", vocab_size=300)
    for (p0, p1) in tok.merges:
        b0, b1 = tok.vocab.get(p0, b""), tok.vocab.get(p1, b"")
        # crude check: a merge shouldn't straddle a letter and '.' / '!'
        if b0 and b1:
            assert not (b0[-1:].isalpha() and b1[:1] in (b".", b"!"))


def test_special_tokens():
    tok = RegexTokenizer()
    tok.train(SAMPLE, vocab_size=300)
    eot_id = 300
    tok.register_special_tokens({"<|endoftext|>": eot_id})

    text = "hello<|endoftext|>world"
    ids = tok.encode(text, allowed_special="all")
    assert eot_id in ids
    assert tok.decode(ids) == text


def test_save_and_load(tmp_path=None):
    import tempfile
    tok = RegexTokenizer()
    tok.train(SAMPLE, vocab_size=300)
    tok.register_special_tokens({"<|endoftext|>": 300})

    with tempfile.TemporaryDirectory() as d:
        prefix = os.path.join(d, "test_model")
        tok.save(prefix)

        loaded = RegexTokenizer()
        loaded.load(f"{prefix}.model")

        assert loaded.merges == tok.merges
        assert loaded.special_tokens == tok.special_tokens
        text = "The quick brown fox<|endoftext|>"
        assert loaded.decode(loaded.encode(text, allowed_special="all")) == text


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASSED: {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
