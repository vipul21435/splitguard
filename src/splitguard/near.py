"""Near-duplicate pairs by MinHash signatures and LSH banding, verified by exact Jaccard.

Every pair returned has an exact Jaccard similarity (on the hashed shingle sets) at or above
the threshold, so there are no false positives. A pair at the threshold is missed with
probability about ``1 - min_recall`` from the banding (the band count is chosen for that),
plus a negligible amount from the signature pre-filter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .text import jaccard

_MASK32 = np.uint64(0xFFFFFFFF)


@dataclass(frozen=True)
class NearPairs:
    """Pairs ``(left[k], right[k])`` with ``left < right`` and their Jaccard ``sim[k]``."""

    left: np.ndarray
    right: np.ndarray
    sim: np.ndarray

    def __len__(self) -> int:
        return len(self.left)

    def as_set(self) -> set[tuple[int, int]]:
        return {(int(a), int(b)) for a, b in zip(self.left, self.right)}


def choose_bands(threshold: float, num_perm: int, min_recall: float = 0.99) -> tuple[int, int]:
    """Pick ``(bands, rows)`` with ``bands * rows <= num_perm``.

    A pair with Jaccard ``s`` becomes a candidate with probability ``1 - (1 - s**rows)**bands``.
    The largest ``rows`` whose probability at ``s = threshold`` is still ``>= min_recall`` is
    chosen: it keeps recall at the threshold while producing the fewest low-similarity
    candidates.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    best = (num_perm, 1)
    for rows in range(1, num_perm + 1):
        bands = num_perm // rows
        if 1.0 - (1.0 - threshold**rows) ** bands >= min_recall:
            best = (bands, rows)
    return best


def minhash_signatures(
    sets: Sequence[np.ndarray], num_perm: int = 128, seed: int = 0, chunk: int = 1 << 15
) -> np.ndarray:
    """MinHash signatures, one row of ``num_perm`` uint32 values per set.

    The hash family is multiply-add modulo 2**64 keeping the high 32 bits,
    ``h(x) = ((a * x + b) mod 2**64) >> 32`` with random odd ``a``. Empty sets get a row of
    the maximum value and should not be compared.
    """
    rng = np.random.default_rng(seed)
    top = np.iinfo(np.uint64).max
    a = rng.integers(0, top, size=num_perm, dtype=np.uint64, endpoint=True) | np.uint64(1)
    b = rng.integers(0, top, size=num_perm, dtype=np.uint64, endpoint=True)
    out = np.full((len(sets), num_perm), np.iinfo(np.uint32).max, dtype=np.uint32)
    batch_rows: list[int] = []
    batch_sets: list[np.ndarray] = []
    total = 0

    def flush() -> None:
        flat = np.concatenate(batch_sets).astype(np.uint64)
        offsets = np.cumsum([0] + [len(s) for s in batch_sets[:-1]])
        hashed = (a[:, None] * flat[None, :] + b[:, None]) >> np.uint64(32)
        mins = np.minimum.reduceat(hashed, offsets, axis=1)
        out[batch_rows] = (mins & _MASK32).astype(np.uint32).T

    for i, s in enumerate(sets):
        if len(s) == 0:
            continue
        batch_rows.append(i)
        batch_sets.append(s)
        total += len(s)
        if total >= chunk:
            flush()
            batch_rows, batch_sets, total = [], [], 0
    if batch_rows:
        flush()
    return out


def _band_keys(sig: np.ndarray, start: int, rows: int) -> np.ndarray:
    key = np.zeros(sig.shape[0], dtype=np.uint64)
    mult = np.uint64(0x100000001B3)
    for c in range(start, start + rows):
        key = (key * mult) ^ sig[:, c].astype(np.uint64)
    return key


def _pairs_in_runs(order: np.ndarray, keys_sorted: np.ndarray) -> list[np.ndarray]:
    """Encode every pair of items that share a key, as ``i * n + j`` with ``i < j``."""
    n_items = len(order)
    boundaries = np.flatnonzero(np.diff(keys_sorted)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n_items]))
    found = []
    for lo, hi in zip(starts, ends):
        m = hi - lo
        if m < 2:
            continue
        members = order[lo:hi]
        ii, jj = np.triu_indices(m, 1)
        x, y = members[ii], members[jj]
        found.append(np.minimum(x, y) * n_items + np.maximum(x, y))
    return found


def near_duplicate_pairs(
    sets: Sequence[np.ndarray],
    threshold: float = 0.8,
    *,
    num_perm: int = 128,
    seed: int = 0,
    min_recall: float = 0.99,
) -> NearPairs:
    """All pairs of sets with Jaccard ``>= threshold``, found through MinHash LSH.

    Empty sets are never paired. Candidates from the bands are first screened by the share
    of agreeing signature values (dropping those more than four standard errors below the
    threshold) and then checked by exact Jaccard.
    """
    n = len(sets)
    empty = NearPairs(np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float64))
    if n < 2:
        return empty
    sig = minhash_signatures(sets, num_perm=num_perm, seed=seed)
    live = np.flatnonzero(np.array([len(s) > 0 for s in sets]))
    if len(live) < 2:
        return empty
    bands, rows = choose_bands(threshold, num_perm, min_recall)
    codes: list[np.ndarray] = []
    sub = sig[live]
    for band in range(bands):
        keys = _band_keys(sub, band * rows, rows)
        order = np.argsort(keys, kind="stable")
        for enc in _pairs_in_runs(order, keys[order]):
            codes.append(enc)
    if not codes:
        return empty
    enc = np.unique(np.concatenate(codes))
    m = len(live)
    left, right = live[enc // m], live[enc % m]
    margin = 4.0 * np.sqrt(threshold * (1.0 - threshold) / num_perm) + 1.0 / num_perm
    keep = np.zeros(len(left), dtype=bool)
    step = 1 << 16
    for lo in range(0, len(left), step):
        hi = lo + step
        est = (sig[left[lo:hi]] == sig[right[lo:hi]]).mean(axis=1)
        keep[lo:hi] = est >= threshold - margin
    left, right = left[keep], right[keep]
    sims = np.array([jaccard(sets[i], sets[j]) for i, j in zip(left, right)], dtype=np.float64)
    ok = sims >= threshold - 1e-12
    return NearPairs(left[ok].astype(np.int64), right[ok].astype(np.int64), sims[ok])


def brute_force_pairs(sets: Sequence[np.ndarray], threshold: float = 0.8) -> NearPairs:
    """Reference: exact Jaccard on every pair. Quadratic, for tests and small inputs."""
    left, right, sims = [], [], []
    for i in range(len(sets)):
        if len(sets[i]) == 0:
            continue
        for j in range(i + 1, len(sets)):
            if len(sets[j]) == 0:
                continue
            s = jaccard(sets[i], sets[j])
            if s >= threshold - 1e-12:
                left.append(i)
                right.append(j)
                sims.append(s)
    return NearPairs(
        np.array(left, dtype=np.int64), np.array(right, dtype=np.int64), np.array(sims, float)
    )
