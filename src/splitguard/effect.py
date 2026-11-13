"""Measured effect of a leak on one weak baseline.

The classifier of :mod:`splitguard.model` is trained and scored twice:

* on the original split, with the test split scored as a whole and separately on its
  *leaked* records (linked by any configured kind to a training record) and its *clean*
  records;
* on a leak-free re-split of the same records (train and test pooled, whole components
  assigned by :func:`splitguard.resplit.resplit`, same sizes as before).

Accuracies carry 95 % Wilson score intervals. Differences between two accuracies carry the
Newcombe hybrid score interval built from the two Wilson intervals; it treats the two
groups as independent samples, which holds for leaked against clean records of one test
split and is only an approximation for the original against the re-split test set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .audit import AuditConfig, audit, build_links
from .data import Dataset, subset
from .graph import link_components
from .model import LogisticRegression, featurize
from .resplit import resplit

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a proportion ``k / n``."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def newcombe(k1: int, n1: int, k2: int, n2: int, z: float = Z95) -> tuple[float, float]:
    """Hybrid score interval for ``k1/n1 - k2/n2`` from two Wilson intervals."""
    if n1 == 0 or n2 == 0:
        return (-1.0, 1.0)
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    return (
        d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
        d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2),
    )


def _score(correct: np.ndarray) -> dict[str, Any]:
    k, n = int(correct.sum()), int(len(correct))
    lo, hi = wilson(k, n)
    return {"n": n, "correct": k, "accuracy": k / n if n else None, "ci95": [lo, hi]}


def _diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    if not a["n"] or not b["n"]:
        return {"difference": None, "ci95": None}
    lo, hi = newcombe(a["correct"], a["n"], b["correct"], b["n"])
    return {"difference": a["accuracy"] - b["accuracy"], "ci95": [lo, hi]}


def _majority(y: np.ndarray) -> float | None:
    if len(y) == 0:
        return None
    return float(np.bincount(y).max() / len(y))


@dataclass
class EffectConfig:
    train: str | None = None
    test: str | None = None
    C: float = 1.0
    bits: int = 20
    max_n: int = 2
    max_iter: int = 300
    label_weight: float = 1.0
    seed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def measure_effect(
    ds: Dataset, audit_cfg: AuditConfig | None = None, cfg: EffectConfig | None = None
) -> dict[str, Any]:
    """Train and score the baseline under the original and a leak-free split."""
    audit_cfg = AuditConfig() if audit_cfg is None else audit_cfg
    cfg = EffectConfig() if cfg is None else cfg
    names = ds.split_names
    train = cfg.train if cfg.train is not None else names[0]
    test = cfg.test if cfg.test is not None else names[-1]
    if train == test or train not in names or test not in names:
        raise ValueError(f"pick two different splits out of {names}")
    s_train, s_test = names.index(train), names.index(test)
    keep = [
        i for i in range(len(ds)) if ds.split[i] in (s_train, s_test) and ds.labels[i] is not None
    ]
    sub = subset(ds, keep)
    tr_id, te_id = sub.split_names.index(train), sub.split_names.index(test)
    label_names = sorted({x for x in sub.labels if x is not None})
    if len(label_names) < 2:
        raise ValueError("the effect needs labels with at least two classes")
    y = np.array([label_names.index(x) for x in sub.labels], dtype=np.int64)
    split = np.array(sub.split, dtype=np.int64)

    links = build_links(sub, audit_cfg)
    rep = audit(sub, audit_cfg, links=links)
    is_train, is_test = split == tr_id, split == te_id
    leaked = np.zeros(len(sub), dtype=bool)
    leaked[rep.leaked(te_id, tr_id)] = True

    X = featurize(sub.values, bits=cfg.bits, max_n=cfg.max_n)

    def fit_score(train_mask: np.ndarray, test_mask: np.ndarray) -> tuple[np.ndarray, int]:
        model = LogisticRegression(C=cfg.C, max_iter=cfg.max_iter)
        tr, te = np.flatnonzero(train_mask), np.flatnonzero(test_mask)
        model.fit(X.take(tr), y[tr])
        return model.predict(X.take(te)) == y[te], model.n_iter_

    correct, it_orig = fit_score(is_train, is_test)
    leaked_in_test = leaked[is_test]
    original = {
        "test": _score(correct),
        "leaked": _score(correct[leaked_in_test]),
        "clean": _score(correct[~leaked_in_test]),
        "majority_class_rate": _majority(y[is_test]),
        "iterations": it_orig,
    }
    original["leaked_minus_clean"] = _diff(original["leaked"], original["clean"])

    comps = link_components(len(sub), links)
    targets = {"train": float(is_train.sum()), "test": float(is_test.sum())}
    new = resplit(comps, targets, sub.labels, label_weight=cfg.label_weight, seed=cfg.seed)
    new_train, new_test = new.assign == 0, new.assign == 1
    correct_new, it_new = fit_score(new_train, new_test)
    resplit_d = {
        "train_records": int(new_train.sum()),
        "test": _score(correct_new),
        "majority_class_rate": _majority(y[new_test]),
        "iterations": it_new,
        "size_error": new.to_dict()["max_size_error"],
    }
    return {
        "train": train,
        "test": test,
        "records": {"train": int(is_train.sum()), "test": int(is_test.sum())},
        "test_leaked_records": int(leaked_in_test.sum()),
        "test_leaked_share": float(leaked_in_test.mean()) if is_test.any() else 0.0,
        "original": original,
        "resplit": resplit_d,
        "original_minus_resplit": _diff(original["test"], resplit_d["test"]),
        "settings": {**audit_cfg.as_dict(), **cfg.as_dict()},
        "note": "one weak baseline: hashed word 1-2-gram logistic regression",
    }
