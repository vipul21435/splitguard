"""The record graph: records as nodes, links of any kind as edges, and its components.

Components come from a disjoint-set forest (union by size, path halving), so a group of
``m`` linked records costs ``m - 1`` unions rather than ``m * (m - 1) / 2`` edges.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .audit import Links


class DisjointSet:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return ra

    def union_chain(self, members: Sequence[int]) -> None:
        """Join every item of ``members`` into one set (a path is enough for connectivity)."""
        for k in range(1, len(members)):
            self.union(members[0], members[k])

    def labels(self) -> np.ndarray:
        """Component label per item, numbered 0.. in order of first appearance."""
        roots = [self.find(x) for x in range(len(self.parent))]
        seen: dict[int, int] = {}
        return np.array([seen.setdefault(r, len(seen)) for r in roots], dtype=np.int64)


def connected_components(n: int, edges: Iterable[tuple[int, int]]) -> np.ndarray:
    dsu = DisjointSet(n)
    for a, b in edges:
        dsu.union(a, b)
    return dsu.labels()


def link_components(n: int, links: Iterable[Links]) -> np.ndarray:
    """Component label per record when every link of every kind is an edge."""
    dsu = DisjointSet(n)
    for lk in links:
        rep = np.full(lk.n_classes, -1, dtype=np.int64)
        for r, c in zip(lk.rec.tolist(), lk.cls.tolist()):
            if rep[c] < 0:
                rep[c] = r
            else:
                dsu.union(int(rep[c]), r)
        for u, v in zip(lk.edge_u.tolist(), lk.edge_v.tolist()):
            if rep[u] >= 0 and rep[v] >= 0:
                dsu.union(int(rep[u]), int(rep[v]))
    return dsu.labels()


@dataclass
class ComponentSummary:
    """Components of the record graph and how they sit across the splits."""

    labels: np.ndarray
    split: np.ndarray
    split_names: list[str]
    ids: list[str]

    def __post_init__(self) -> None:
        n_comp = int(self.labels.max()) + 1 if len(self.labels) else 0
        self.counts = np.zeros((n_comp, len(self.split_names)), dtype=np.int64)
        np.add.at(self.counts, (self.labels, self.split), 1)
        self.sizes = self.counts.sum(axis=1)
        self.crossing = np.flatnonzero((self.counts > 0).sum(axis=1) >= 2)

    def members(self, comp: int) -> np.ndarray:
        return np.flatnonzero(self.labels == comp)

    def to_dict(self, examples: int = 5, max_ids: int = 10) -> dict[str, Any]:
        sizes = self.sizes
        cross = self.crossing
        n = len(self.labels)
        in_cross = np.isin(self.labels, cross)
        per_split = []
        for s, name in enumerate(self.split_names):
            total = int(np.sum(self.split == s))
            k = int(np.sum(in_cross & (self.split == s)))
            share = k / total if total else 0.0
            per_split.append({"split": name, "records": total, "in_crossing": k, "share": share})
        hist: dict[str, int] = {}
        for lo, hi in ((1, 1), (2, 2), (3, 5), (6, 20), (21, 100), (101, None)):
            sel = (sizes >= lo) & ((sizes <= hi) if hi is not None else True)
            name = f"{lo}" if lo == hi else (f"{lo}+" if hi is None else f"{lo}-{hi}")
            hist[name] = int(np.sum(sel))
        order = cross[np.argsort(-sizes[cross], kind="stable")]
        top = []
        for c in order[:examples].tolist():
            mem = self.members(c)
            top.append(
                {
                    "component": c,
                    "size": int(sizes[c]),
                    "by_split": {
                        name: int(self.counts[c, s]) for s, name in enumerate(self.split_names)
                    },
                    "ids": [self.ids[i] for i in mem[:max_ids].tolist()],
                }
            )
        return {
            "records": n,
            "components": int(len(sizes)),
            "singletons": int(np.sum(sizes == 1)),
            "largest": int(sizes.max()) if len(sizes) else 0,
            "size_histogram": hist,
            "crossing": {
                "components": int(len(cross)),
                "records": int(sizes[cross].sum()),
                "largest": int(sizes[cross].max()) if len(cross) else 0,
                "by_split": per_split,
            },
            "largest_crossing": top,
        }
