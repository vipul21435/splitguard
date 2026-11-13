import numpy as np
import pytest

from splitguard.near import (
    brute_force_pairs,
    choose_bands,
    minhash_signatures,
    near_duplicate_pairs,
)
from splitguard.text import jaccard, shingles

WORDS = [f"w{i}" for i in range(400)]


def planted_texts(seed: int, n_base: int = 60, copies: int = 3) -> list[str]:
    """Random word texts plus edited copies of each: a word swapped, dropped or added."""
    rng = np.random.default_rng(seed)
    texts = []
    for _ in range(n_base):
        base = list(rng.choice(WORDS, size=int(rng.integers(12, 30))))
        texts.append(" ".join(base))
        for _ in range(copies):
            words = list(base)
            for _ in range(int(rng.integers(0, 3))):
                op = rng.integers(0, 3)
                k = int(rng.integers(0, len(words)))
                if op == 0:
                    words[k] = str(rng.choice(WORDS))
                elif op == 1 and len(words) > 2:
                    del words[k]
                else:
                    words.insert(k, str(rng.choice(WORDS)))
            texts.append(" ".join(words))
    order = rng.permutation(len(texts))
    return [texts[i] for i in order]


def test_choose_bands_meets_recall_at_threshold():
    for t in (0.5, 0.7, 0.8, 0.9):
        for p in (64, 128, 256):
            b, r = choose_bands(t, p, 0.99)
            assert b * r <= p
            assert 1 - (1 - t**r) ** b >= 0.99
    assert choose_bands(0.8, 128) == (21, 6)
    with pytest.raises(ValueError):
        choose_bands(0.0, 128)


def test_minhash_estimates_jaccard():
    a = shingles("the quick brown fox jumps over the lazy dog near the river bank", 3)
    b = shingles("the quick brown fox jumped over the lazy dog near the river bank", 3)
    sig = minhash_signatures([a, b], num_perm=512, seed=3)
    est = (sig[0] == sig[1]).mean()
    assert est == pytest.approx(jaccard(a, b), abs=0.08)
    same = minhash_signatures([a, a], num_perm=64)
    assert np.array_equal(same[0], same[1])


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("unit,n,threshold", [("char", 5, 0.8), ("word", 2, 0.7)])
def test_lsh_matches_brute_force(seed, unit, n, threshold):
    sets = [shingles(t, n, unit) for t in planted_texts(seed)]
    truth = brute_force_pairs(sets, threshold)
    found = near_duplicate_pairs(sets, threshold, seed=seed)
    assert len(truth) > 50
    # no false positives, and similarities agree with the exact values
    truth_sim = dict(zip(zip(truth.left.tolist(), truth.right.tolist()), truth.sim.tolist()))
    for a, b, s in zip(found.left.tolist(), found.right.tolist(), found.sim.tolist()):
        assert a < b
        assert s == pytest.approx(truth_sim[(a, b)], abs=1e-12)
    recall = len(found.as_set() & truth.as_set()) / len(truth)
    assert recall >= 0.97


def test_empty_and_tiny_inputs():
    e = np.empty(0, dtype=np.uint32)
    assert len(near_duplicate_pairs([])) == 0
    assert len(near_duplicate_pairs([e, e, e])) == 0
    x = shingles("same text here", 3)
    pairs = near_duplicate_pairs([x, e, x])
    assert pairs.as_set() == {(0, 2)}
    assert pairs.sim[0] == pytest.approx(1.0)
