"""Leak audit: records of one split that are linked to records of another split.

A *link kind* says when two records are linked:

``exact``
    the whole record is identical after :func:`splitguard.text.normalize` (all text fields).
``near``
    the shingle sets of the whole normalised record have Jaccard similarity at or above the
    threshold; exact copies count here too.
``group:<column>``
    the records carry the same non-empty value in a group column.
``field``
    (records with two or more text fields) one normalised field value of one record equals a
    field value of the other, in any field position, even when the whole records differ.

Every kind is stored the same way: records belong to *classes* (records in one class are all
linked to each other) and classes may be joined by *edges* (every record of one class is
linked to every record of the other). Kinds overlap; ``any`` is their union.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .data import Dataset
from .near import near_duplicate_pairs
from .text import normalize, shingles

FIELD_SEP = "\x1f"


@dataclass
class AuditConfig:
    """Settings for :func:`audit`; the defaults are the CLI defaults."""

    kinds: Sequence[str] = ("exact", "near", "group", "field")
    threshold: float = 0.8
    shingle_n: int = 5
    shingle_unit: str = "char"
    num_perm: int = 128
    seed: int = 0
    min_field_chars: int = 1
    max_examples: int = 5
    show_text: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "kinds": list(self.kinds),
            "threshold": self.threshold,
            "shingle_n": self.shingle_n,
            "shingle_unit": self.shingle_unit,
            "num_perm": self.num_perm,
            "seed": self.seed,
            "min_field_chars": self.min_field_chars,
        }


@dataclass
class Links:
    """Records linked by one kind: memberships ``(rec[k], cls[k])`` and class edges."""

    kind: str
    rec: np.ndarray
    cls: np.ndarray
    keys: list[str]
    edge_u: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    edge_v: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    edge_sim: np.ndarray = field(default_factory=lambda: np.empty(0, np.float64))

    @property
    def n_classes(self) -> int:
        return len(self.keys)


def _classes_from_keys(kind: str, keys_per_record: Sequence[Sequence[str]]) -> Links:
    index: dict[str, int] = {}
    rec: list[int] = []
    cls: list[int] = []
    for r, keys in enumerate(keys_per_record):
        seen = set()
        for key in keys:
            c = index.setdefault(key, len(index))
            if c not in seen:
                seen.add(c)
                rec.append(r)
                cls.append(c)
    return Links(kind, np.array(rec, np.int64), np.array(cls, np.int64), list(index))


def record_keys(ds: Dataset) -> list[str]:
    """Normalised whole-record key per record ("" when every field is empty)."""
    out = []
    for values in ds.values:
        parts = [normalize(v) for v in values]
        out.append(FIELD_SEP.join(parts) if any(parts) else "")
    return out


def exact_links(ds: Dataset, keys: Sequence[str] | None = None) -> Links:
    keys = record_keys(ds) if keys is None else keys
    return _classes_from_keys("exact", [[k] if k else [] for k in keys])


def near_links(ds: Dataset, cfg: AuditConfig, exact: Links | None = None) -> Links:
    """Near-duplicate links: LSH over the distinct normalised records, then exact Jaccard."""
    exact = exact_links(ds) if exact is None else exact
    sets = [
        shingles(key.replace(FIELD_SEP, " "), cfg.shingle_n, cfg.shingle_unit)
        for key in exact.keys
    ]
    pairs = near_duplicate_pairs(
        sets, cfg.threshold, num_perm=cfg.num_perm, seed=cfg.seed
    )
    return Links("near", exact.rec, exact.cls, exact.keys, pairs.left, pairs.right, pairs.sim)


def group_links(ds: Dataset, column: str) -> Links:
    values = ds.groups[column]
    return _classes_from_keys(f"group:{column}", [[v] if v is not None else [] for v in values])


def field_links(ds: Dataset, min_chars: int = 1) -> Links:
    keys_per_record = []
    for values in ds.values:
        keys = [normalize(v) for v in values]
        keys_per_record.append([k for k in keys if len(k) >= max(min_chars, 1)])
    return _classes_from_keys("field", keys_per_record)


def build_links(ds: Dataset, cfg: AuditConfig) -> list[Links]:
    """The links of every kind in ``cfg.kinds`` that applies to ``ds``."""
    unknown = set(cfg.kinds) - {"exact", "near", "group", "field"}
    if unknown:
        raise ValueError(f"unknown link kind(s) {sorted(unknown)}")
    out: list[Links] = []
    exact = None
    if "exact" in cfg.kinds or "near" in cfg.kinds:
        exact = exact_links(ds)
    if "exact" in cfg.kinds:
        out.append(exact)
    if "near" in cfg.kinds:
        out.append(near_links(ds, cfg, exact))
    if "group" in cfg.kinds:
        out.extend(group_links(ds, g) for g in ds.groups)
    if "field" in cfg.kinds and len(ds.fields) >= 2:
        out.append(field_links(ds, cfg.min_field_chars))
    return out


def cross_bits(links: Links, split: np.ndarray, n: int) -> np.ndarray:
    """Per record, a bit mask of the *other* splits it has a linked record in."""
    own = np.left_shift(np.int64(1), split)
    presence = np.zeros(links.n_classes, np.int64)
    np.bitwise_or.at(presence, links.cls, own[links.rec])
    reach = presence.copy()
    if len(links.edge_u):
        np.bitwise_or.at(reach, links.edge_u, presence[links.edge_v])
        np.bitwise_or.at(reach, links.edge_v, presence[links.edge_u])
    bits = np.zeros(n, np.int64)
    np.bitwise_or.at(bits, links.rec, reach[links.cls])
    return bits & ~own


@dataclass
class KindResult:
    """Audit numbers for one link kind."""

    kind: str
    bits: np.ndarray
    pairs: np.ndarray  # [s, t]: linked record pairs between split s and split t
    agree: np.ndarray  # [s, t]: of those with both labels known, how many agree
    labelled: np.ndarray
    examples: list[dict[str, Any]]


def _pair_counts(
    links: Links, split: np.ndarray, label_ids: np.ndarray, n_splits: int, n_labels: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros((links.n_classes, n_splits), np.int64)
    np.add.at(counts, (links.cls, split[links.rec]), 1)
    known = label_ids[links.rec] >= 0
    lab = np.zeros((links.n_classes, n_splits, max(n_labels, 1)), np.int64)
    np.add.at(
        lab,
        (links.cls[known], split[links.rec[known]], label_ids[links.rec[known]]),
        1,
    )
    lab_tot = lab.sum(axis=2)
    pairs = np.zeros((n_splits, n_splits), np.int64)
    agree = np.zeros_like(pairs)
    labelled = np.zeros_like(pairs)
    u, v = links.edge_u, links.edge_v
    for s in range(n_splits):
        for t in range(n_splits):
            if s == t:
                continue
            p = int(np.dot(counts[:, s], counts[:, t]))
            a = int(np.sum(lab[:, s, :] * lab[:, t, :]))
            m = int(np.dot(lab_tot[:, s], lab_tot[:, t]))
            if len(u):
                p += int(np.dot(counts[u, s], counts[v, t]) + np.dot(counts[v, s], counts[u, t]))
                a += int(
                    np.sum(lab[u, s, :] * lab[v, t, :]) + np.sum(lab[v, s, :] * lab[u, t, :])
                )
                m += int(
                    np.dot(lab_tot[u, s], lab_tot[v, t]) + np.dot(lab_tot[v, s], lab_tot[u, t])
                )
            pairs[s, t], agree[s, t], labelled[s, t] = p, a, m
    return pairs, agree, labelled


def _examples(links: Links, ds: Dataset, split: np.ndarray, cfg: AuditConfig) -> list[dict]:
    if cfg.max_examples <= 0:
        return []
    order = np.argsort(links.cls, kind="stable")
    rec_sorted, cls_sorted = links.rec[order], links.cls[order]
    starts = np.searchsorted(cls_sorted, np.arange(links.n_classes + 1))

    def members(c: int) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for r in rec_sorted[starts[c] : starts[c + 1]].tolist():
            out.setdefault(ds.split_names[split[r]], []).append(ds.ids[r])
        return out

    def show(c: int) -> str:
        return links.keys[c].replace(FIELD_SEP, " | ")[:120]

    own = np.left_shift(np.int64(1), split)
    presence = np.zeros(links.n_classes, np.int64)
    np.bitwise_or.at(presence, links.cls, own[links.rec])
    found: list[dict] = []
    if len(links.edge_u):
        pu, pv = presence[links.edge_u], presence[links.edge_v]
        # an edge links two splits unless both classes sit in the same single split
        crossing = ~((pu == pv) & ((pu & (pu - 1)) == 0))
        idx = np.flatnonzero(crossing)
        idx = idx[np.argsort(-links.edge_sim[idx], kind="stable")]
        for k in idx[: cfg.max_examples].tolist():
            u, v = int(links.edge_u[k]), int(links.edge_v[k])
            item: dict[str, Any] = {
                "similarity": round(float(links.edge_sim[k]), 4),
                "a": members(u),
                "b": members(v),
            }
            if cfg.show_text:
                item["a_text"], item["b_text"] = show(u), show(v)
            found.append(item)
        return found
    sizes = np.diff(starts)
    multi = np.flatnonzero((presence & (presence - 1)) != 0)
    multi = multi[np.argsort(-sizes[multi], kind="stable")]
    for c in multi[: cfg.max_examples].tolist():
        item = {"members": members(c)}
        if cfg.show_text:
            item["key"] = show(c)
        found.append(item)
    return found


@dataclass
class AuditReport:
    split_names: list[str]
    sizes: list[int]
    results: list[KindResult]
    any_bits: np.ndarray
    split: np.ndarray
    config: AuditConfig

    def leaked(self, target: int, source: int, kind: str | None = None) -> np.ndarray:
        """Indices of records of split ``target`` linked to some record of split ``source``."""
        bits = self.any_bits if kind is None else self.result(kind).bits
        mask = (self.split == target) & ((bits >> source) & 1).astype(bool)
        return np.flatnonzero(mask)

    def result(self, kind: str) -> KindResult:
        for r in self.results:
            if r.kind == kind:
                return r
        raise KeyError(kind)

    def to_dict(self) -> dict[str, Any]:
        def by_pair(bits: np.ndarray, res: KindResult | None) -> list[dict[str, Any]]:
            rows = []
            for t, tname in enumerate(self.split_names):
                for s, sname in enumerate(self.split_names):
                    if s == t:
                        continue
                    k = int(np.sum((self.split == t) & (((bits >> s) & 1) == 1)))
                    row: dict[str, Any] = {
                        "split": tname,
                        "linked_to": sname,
                        "records": k,
                        "share": k / self.sizes[t] if self.sizes[t] else 0.0,
                    }
                    if res is not None:
                        row["pairs"] = int(res.pairs[s, t])
                        lab = int(res.labelled[s, t])
                        row["labelled_pairs"] = lab
                        row["label_agreement"] = int(res.agree[s, t]) / lab if lab else None
                    rows.append(row)
            return rows

        return {
            "splits": [{"name": n, "records": k} for n, k in zip(self.split_names, self.sizes)],
            "settings": self.config.as_dict(),
            "kinds": [
                {"kind": r.kind, "by_split": by_pair(r.bits, r), "examples": r.examples}
                for r in self.results
            ],
            "any": {"by_split": by_pair(self.any_bits, None)},
        }


def label_index(labels: Sequence[str | None]) -> tuple[np.ndarray, list[str]]:
    names = sorted({x for x in labels if x is not None})
    pos = {x: i for i, x in enumerate(names)}
    return np.array([pos[x] if x is not None else -1 for x in labels], np.int64), names


def audit(
    ds: Dataset, cfg: AuditConfig | None = None, links: list[Links] | None = None
) -> AuditReport:
    """Run every configured link kind over ``ds`` and count cross-split links."""
    cfg = AuditConfig() if cfg is None else cfg
    if len(ds.split_names) > 62:
        raise ValueError("at most 62 splits")
    links = build_links(ds, cfg) if links is None else links
    split = np.array(ds.split, np.int64)
    n = len(ds)
    label_ids, label_names = label_index(ds.labels)
    results = []
    any_bits = np.zeros(n, np.int64)
    for lk in links:
        bits = cross_bits(lk, split, n)
        any_bits |= bits
        pairs, agree, labelled = _pair_counts(
            lk, split, label_ids, len(ds.split_names), len(label_names)
        )
        results.append(
            KindResult(lk.kind, bits, pairs, agree, labelled, _examples(lk, ds, split, cfg))
        )
    sizes = [int(np.sum(split == s)) for s in range(len(ds.split_names))]
    return AuditReport(list(ds.split_names), sizes, results, any_bits, split, cfg)
