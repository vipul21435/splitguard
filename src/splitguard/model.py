"""A small text classifier: hashed bags of word n-grams and L2 logistic regression.

It is one weak baseline, kept here so that the effect of a leak can be measured without
extra dependencies. Features are word 1- and 2-grams of each normalised text field, hashed
into ``2**bits`` buckets with CRC-32 and prefixed by the field position, plus (for records
with two or more fields) the n-grams that appear in every field, which lets a pair model see
overlap. Rows are L2-normalised binary indicators.

Two classes use the logistic loss on one weight vector, more use the softmax loss. The
objective is ``C * sum_i loss_i + ||W||**2 / 2`` with an unpenalised intercept (the
convention of scikit-learn's ``LogisticRegression``), minimised by limited-memory BFGS
(the two-loop recursion) with a backtracking Armijo line search.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .text import normalize


@dataclass
class SparseRows:
    """Row-compressed sparse matrix (``indptr``, ``indices``, ``data``)."""

    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    n_cols: int

    def __post_init__(self) -> None:
        self.rows = np.repeat(np.arange(len(self.indptr) - 1), np.diff(self.indptr))

    @property
    def n_rows(self) -> int:
        return len(self.indptr) - 1

    def take(self, idx: np.ndarray) -> SparseRows:
        idx = np.asarray(idx, dtype=np.int64)
        lens = np.diff(self.indptr)[idx]
        indptr = np.concatenate(([0], np.cumsum(lens)))
        starts = self.indptr[idx]
        pos = np.repeat(starts - indptr[:-1], lens) + np.arange(indptr[-1])
        return SparseRows(indptr, self.indices[pos], self.data[pos], self.n_cols)

    def dot(self, w: np.ndarray) -> np.ndarray:
        """``X @ w`` for ``w`` of shape (n_cols,) or (n_cols, k)."""
        if w.ndim == 1:
            return np.bincount(self.rows, self.data * w[self.indices], minlength=self.n_rows)
        return np.stack([self.dot(w[:, j]) for j in range(w.shape[1])], axis=1)

    def tdot(self, r: np.ndarray) -> np.ndarray:
        """``X.T @ r`` for ``r`` of shape (n_rows,) or (n_rows, k)."""
        if r.ndim == 1:
            return np.bincount(self.indices, self.data * r[self.rows], minlength=self.n_cols)
        return np.stack([self.tdot(r[:, j]) for j in range(r.shape[1])], axis=1)

    def toarray(self) -> np.ndarray:
        out = np.zeros((self.n_rows, self.n_cols))
        np.add.at(out, (self.rows, self.indices), self.data)
        return out


def _grams(text: str, max_n: int) -> list[str]:
    words = normalize(text).split()
    out = []
    for n in range(1, max_n + 1):
        out.extend(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))
    return out


def featurize(
    records: Sequence[Sequence[str]], bits: int = 20, max_n: int = 2, shared: bool = True
) -> SparseRows:
    """Hashed, L2-normalised binary n-gram features, one row per record."""
    n_cols = 1 << bits
    mask = n_cols - 1
    indptr = [0]
    indices: list[int] = []
    for fields in records:
        feats: set[int] = set()
        per_field = [set(_grams(t, max_n)) for t in fields]
        for f, grams in enumerate(per_field):
            feats.update(zlib.crc32(f"{f}\x00{g}".encode()) & mask for g in grams)
        if shared and len(per_field) >= 2:
            common = set.intersection(*per_field)
            feats.update(zlib.crc32(f"&\x00{g}".encode()) & mask for g in common)
        row = sorted(feats)
        indices.extend(row)
        indptr.append(len(indices))
    ip = np.array(indptr, dtype=np.int64)
    lens = np.diff(ip)
    data = np.repeat(1.0 / np.sqrt(np.maximum(lens, 1)), lens)
    return SparseRows(ip, np.array(indices, dtype=np.int64), data, n_cols)


def lbfgs(
    fun: Callable[[np.ndarray], tuple[float, np.ndarray]],
    x0: np.ndarray,
    *,
    max_iter: int = 500,
    memory: int = 10,
    gtol: float = 1e-6,
) -> tuple[np.ndarray, int]:
    """Minimise a smooth convex function; returns ``(x, iterations)``.

    Stops when the largest gradient entry falls below ``gtol`` times its starting value.
    """
    x = x0.astype(np.float64).copy()
    f, g = fun(x)
    stop = gtol * max(1.0, float(np.max(np.abs(g))))
    s_hist: list[np.ndarray] = []
    y_hist: list[np.ndarray] = []
    it = 0
    while it < max_iter and float(np.max(np.abs(g))) > stop:
        it += 1
        q = g.copy()
        alphas = []
        for s, y in zip(reversed(s_hist), reversed(y_hist)):
            rho = 1.0 / float(y @ s)
            a = rho * float(s @ q)
            q -= a * y
            alphas.append((rho, a))
        if s_hist:
            q *= float(s_hist[-1] @ y_hist[-1]) / float(y_hist[-1] @ y_hist[-1])
        else:
            q /= max(1.0, float(np.linalg.norm(g)))
        for (s, y), (rho, a) in zip(zip(s_hist, y_hist), reversed(alphas)):
            b = rho * float(y @ q)
            q += (a - b) * s
        d = -q
        slope = float(g @ d)
        if slope >= 0:
            d, slope = -g, -float(g @ g)
            s_hist.clear()
            y_hist.clear()
        t = 1.0
        while True:
            x_new = x + t * d
            f_new, g_new = fun(x_new)
            if f_new <= f + 1e-4 * t * slope or t < 1e-12:
                break
            t *= 0.5
        s_vec, y_vec = x_new - x, g_new - g
        if float(s_vec @ y_vec) > 1e-12:
            s_hist.append(s_vec)
            y_hist.append(y_vec)
            if len(s_hist) > memory:
                s_hist.pop(0)
                y_hist.pop(0)
        x, f, g = x_new, f_new, g_new
    return x, it


@dataclass
class LogisticRegression:
    """L2-penalised logistic (two classes) or softmax (more) regression on sparse rows."""

    C: float = 1.0
    max_iter: int = 500
    gtol: float = 1e-6

    def fit(self, X: SparseRows, y: Sequence[int] | np.ndarray) -> LogisticRegression:
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y)
        k = len(self.classes_)
        if k < 2:
            raise ValueError("need at least two classes")
        yi = np.searchsorted(self.classes_, y)
        d = X.n_cols
        C = self.C
        if k == 2:
            t = yi.astype(np.float64)

            def fun(w: np.ndarray) -> tuple[float, np.ndarray]:
                z = X.dot(w[:d]) + w[d]
                # log(1 + exp(-z)) for t = 1 and log(1 + exp(z)) for t = 0
                loss = np.logaddexp(0.0, np.where(t > 0, -z, z)).sum()
                p = 0.5 * (1.0 + np.tanh(0.5 * z))
                r = p - t
                grad = np.empty(d + 1)
                grad[:d] = C * X.tdot(r) + w[:d]
                grad[d] = C * r.sum()
                return C * loss + 0.5 * float(w[:d] @ w[:d]), grad

            w, self.n_iter_ = lbfgs(fun, np.zeros(d + 1), max_iter=self.max_iter, gtol=self.gtol)
            self.coef_ = w[:d][None, :]
            self.intercept_ = w[d:]
        else:
            onehot = np.zeros((X.n_rows, k))
            onehot[np.arange(X.n_rows), yi] = 1.0

            def fun(w: np.ndarray) -> tuple[float, np.ndarray]:
                W = w[: d * k].reshape(d, k)
                b = w[d * k :]
                z = X.dot(W) + b
                zmax = z.max(axis=1, keepdims=True)
                lse = zmax[:, 0] + np.log(np.exp(z - zmax).sum(axis=1))
                loss = float(np.sum(lse - z[np.arange(X.n_rows), yi]))
                p = np.exp(z - lse[:, None])
                r = p - onehot
                grad = np.concatenate([(C * X.tdot(r) + W).ravel(), C * r.sum(axis=0)])
                return C * loss + 0.5 * float(np.sum(W * W)), grad

            w, self.n_iter_ = lbfgs(
                fun, np.zeros(d * k + k), max_iter=self.max_iter, gtol=self.gtol
            )
            self.coef_ = w[: d * k].reshape(d, k).T
            self.intercept_ = w[d * k :]
        return self

    def decision_function(self, X: SparseRows) -> np.ndarray:
        z = X.dot(self.coef_.T) + self.intercept_
        return z[:, 0] if z.shape[1] == 1 else z

    def predict_proba(self, X: SparseRows) -> np.ndarray:
        z = self.decision_function(X)
        if z.ndim == 1:
            p1 = 0.5 * (1.0 + np.tanh(0.5 * z))
            return np.stack([1.0 - p1, p1], axis=1)
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X: SparseRows) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
