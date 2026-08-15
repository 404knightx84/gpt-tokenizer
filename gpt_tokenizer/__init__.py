"""
gpt_tokenizer
=============
A from-scratch, byte-level BPE tokenizer library, GPT-2 / GPT-4 style.

Public API:
    BasicTokenizer  - pure byte-level BPE, no pre-splitting (simplest, for learning)
    RegexTokenizer  - + regex pre-tokenization + special-token support (production-style)
"""

from .basic import BasicTokenizer
from .regex_tokenizer import RegexTokenizer

__all__ = ["BasicTokenizer", "RegexTokenizer"]
__version__ = "0.1.0"
