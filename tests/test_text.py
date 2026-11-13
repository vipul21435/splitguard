import numpy as np
import pytest

from splitguard.text import jaccard, normalize, shingles


def test_normalize_case_punctuation_whitespace_and_width():
    assert normalize("  Hello,   WORLD!! ") == "hello world"
    assert normalize("snake_case-name") == "snake case name"
    assert normalize("ＡＢＣ　１２３") == "abc 123"  # full-width forms fold under NFKC
    assert normalize("Straße") == "strasse"
    assert normalize("Hi!", strip_punct=False) == "hi!"
    assert normalize("Hi", casefold=False) == "Hi"


def test_char_and_word_shingles():
    assert len(shingles("abcdef", n=3)) == 4
    assert len(shingles("ab", n=3)) == 1  # short text is one shingle
    assert len(shingles("", n=3)) == 0
    assert len(shingles("a b c d", n=2, unit="word")) == 3
    assert len(shingles("a a a a", n=2, unit="word")) == 1  # a set, not a multiset
    s = shingles("the cat sat", n=2)
    assert s.dtype == np.uint32 and np.all(np.diff(s.astype(np.int64)) > 0)


def test_shingles_are_stable_across_runs():
    # CRC-32 of "abc" is fixed, unlike Python's salted hash().
    assert shingles("abc", n=3).tolist() == [0x352441C2]


def test_bad_shingle_arguments():
    with pytest.raises(ValueError):
        shingles("abc", n=0)
    with pytest.raises(ValueError):
        shingles("abc", unit="byte")


def test_jaccard():
    a = np.array([1, 2, 3], dtype=np.uint32)
    b = np.array([2, 3, 4, 5], dtype=np.uint32)
    assert jaccard(a, b) == pytest.approx(2 / 5)
    assert jaccard(a, a) == pytest.approx(1.0)
    e = np.empty(0, dtype=np.uint32)
    assert jaccard(e, e) == 0.0
    assert jaccard(a, e) == 0.0
