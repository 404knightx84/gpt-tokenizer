"""
patterns.py
-----------
Pre-tokenization regex patterns. These split raw text into chunks *before*
BPE merges are learned or applied, so that merges never bridge across
chunk boundaries (e.g. a word and the punctuation after it never fuse into
one token). Requires the third-party `regex` module (not stdlib `re`)
because \\p{L} / \\p{N} unicode property classes are used.
"""

# GPT-2 style: simpler, ASCII-oriented contraction handling.
GPT2_SPLIT_PATTERN = (
    r"""'s|'t|'re|'ve|'m|'ll|'d"""
    r"""| ?\p{L}+"""
    r"""| ?\p{N}+"""
    r"""| ?[^\s\p{L}\p{N}]+"""
    r"""|\s+(?!\S)"""
    r"""|\s+"""
)

# GPT-4 style: case-insensitive contractions, numbers capped at 3 digits
# per chunk (keeps very long digit runs from becoming single mega-tokens).
GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)"""
    r"""|[^\r\n\p{L}\p{N}]?\p{L}+"""
    r"""|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]+[\r\n]*"""
    r"""|\s*[\r\n]"""
    r"""|\s+(?!\S)"""
    r"""|\s+"""
)
