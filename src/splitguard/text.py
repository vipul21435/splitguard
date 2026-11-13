"""Text normalisation and hashed shingle sets."""

from __future__ import annotations

import re
import unicodedata
import zlib

import numpy as np

_NON_WORD = re.compile(r"[\W_]+")


def normalize(text: str, *, casefold: bool = True, strip_punct: bool = True) -> str:
    """NFKC-normalise, case-fold, turn punctuation into spaces and collapse whitespace."""
    out = unicodedata.normalize("NFKC", text)
    if casefold:
        out = out.casefold()
    if strip_punct:
        out = _NON_WORD.sub(" ", out)
    return " ".join(out.split())


def shingles(text: str, n: int = 5, unit: str = "char") -> np.ndarray:
    """Return the set of ``n``-gram shingles of ``text`` as sorted unique 32-bit hashes.

    ``unit`` is ``"char"`` (character n-grams) or ``"word"`` (whitespace-token n-grams). A text
    shorter than ``n`` units becomes a single shingle; an empty text gives an empty set.
    CRC-32 is used so that the hashes are the same on every platform and run.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if unit == "char":
        units: str | list[str] = text
        joiner = ""
    elif unit == "word":
        units = text.split()
        joiner = " "
    else:
        raise ValueError(f"unit must be 'char' or 'word', got {unit!r}")
    if len(units) == 0:
        return np.empty(0, dtype=np.uint32)
    if len(units) <= n:
        grams = [joiner.join(units)]
    else:
        grams = [joiner.join(units[i : i + n]) for i in range(len(units) - n + 1)]
    hashes = np.fromiter(
        (zlib.crc32(g.encode("utf-8")) for g in grams), dtype=np.uint32, count=len(grams)
    )
    return np.unique(hashes)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard similarity of two sorted unique arrays; 0.0 when both are empty."""
    if len(a) == 0 and len(b) == 0:
        return 0.0
    inter = len(np.intersect1d(a, b, assume_unique=True))
    return inter / (len(a) + len(b) - inter)
