"""Connected components with a disjoint-set forest (union by size, path halving)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


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
