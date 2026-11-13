"""Leak-free re-split: whole components go to one side.

The method is a seeded greedy pass, largest component first, so that the many small
components placed last can even out the sizes and label counts:

1. Each split ``s`` gets a target size ``T_s = f_s * N`` and, when labels are known, label
   targets ``T_sl = f_s * N_l``, where ``N_l`` counts label ``l`` over all records.
2. Components are shuffled with the seed and then sorted by size, largest first (the shuffle
   only breaks ties).
3. Each component, with ``n_c`` records of which ``y_cl`` carry label ``l``, goes to the
   split that adds least to the discrepancy

       D = sum_s (n_s - T_s)**2 / T_s  +  w * sum_s sum_l (n_sl - T_sl)**2 / T_sl,

   that is, the split minimising
   ``(2 (n_s - T_s) n_c + n_c**2) / T_s + w * sum_l (2 (n_sl - T_sl) y_cl + y_cl**2) / T_sl``.
   ``w`` (``label_weight``) is 0 to ignore labels.

A component is never divided, so no link of any kind can cross the new splits. Large
components make the targets unreachable; the result reports how far it ended up.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Resplit:
    """New split per record and how close it came to the targets."""

    names: list[str]
    assign: np.ndarray
    fractions: list[float]
    label_names: list[str]
    label_ids: np.ndarray

    def sizes(self) -> list[int]:
        return [int(np.sum(self.assign == s)) for s in range(len(self.names))]

    def to_dict(self) -> dict[str, Any]:
        n = len(self.assign)
        sizes = self.sizes()
        out: dict[str, Any] = {"records": n, "splits": []}
        known = self.label_ids >= 0
        overall = (
            np.bincount(self.label_ids[known], minlength=len(self.label_names)) / known.sum()
            if known.any()
            else np.zeros(0)
        )
        worst_label = 0.0
        for s, name in enumerate(self.names):
            target = self.fractions[s] * n
            row: dict[str, Any] = {
                "split": name,
                "target_fraction": self.fractions[s],
                "target_records": round(target, 1),
                "records": sizes[s],
                "fraction": sizes[s] / n if n else 0.0,
                "size_error": (sizes[s] - target) / target if target else 0.0,
            }
            sel = known & (self.assign == s)
            if self.label_names and sel.any():
                mix = np.bincount(self.label_ids[sel], minlength=len(self.label_names)) / sel.sum()
                row["label_mix"] = {k: float(v) for k, v in zip(self.label_names, mix)}
                gap = float(np.max(np.abs(mix - overall)))
                row["label_gap"] = gap
                worst_label = max(worst_label, gap)
            out["splits"].append(row)
        if self.label_names:
            out["label_mix_overall"] = {k: float(v) for k, v in zip(self.label_names, overall)}
            out["max_label_gap"] = worst_label
        out["max_size_error"] = max((abs(r["size_error"]) for r in out["splits"]), default=0.0)
        return out


def parse_sizes(text: str) -> dict[str, float]:
    """``"train=0.8,test=0.2"`` (fractions) or ``"train=8000,test=2000"`` (counts)."""
    out: dict[str, float] = {}
    for part in text.split(","):
        name, sep, value = part.partition("=")
        if not sep or not name.strip():
            raise ValueError(f"bad size {part!r}; use name=fraction or name=count")
        out[name.strip()] = float(value)
    if any(v < 0 for v in out.values()) or sum(out.values()) <= 0:
        raise ValueError("sizes must be non-negative and not all zero")
    return out


def normalise_targets(targets: Mapping[str, float]) -> tuple[list[str], list[float]]:
    names = list(targets)
    total = float(sum(targets.values()))
    return names, [targets[k] / total for k in names]


def resplit(
    components: np.ndarray,
    targets: Mapping[str, float],
    labels: Sequence[str | None] | None = None,
    *,
    label_weight: float = 1.0,
    seed: int = 0,
) -> Resplit:
    """Assign whole components to the splits in ``targets`` (fractions or counts)."""
    names, fractions = normalise_targets(targets)
    comps = np.asarray(components, dtype=np.int64)
    n = len(comps)
    labels = [None] * n if labels is None else list(labels)
    if len(labels) != n:
        raise ValueError("labels and components differ in length")
    label_names = sorted({x for x in labels if x is not None})
    pos = {x: i for i, x in enumerate(label_names)}
    label_ids = np.array([pos[x] if x is not None else -1 for x in labels], dtype=np.int64)
    n_comp = int(comps.max()) + 1 if n else 0
    n_lab = len(label_names)

    sizes = np.bincount(comps, minlength=n_comp)
    known = label_ids >= 0
    comp_lab = np.zeros((n_comp, max(n_lab, 1)), dtype=np.int64)
    np.add.at(comp_lab, (comps[known], label_ids[known]), 1)
    lab_total = comp_lab.sum(axis=0)

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_comp)
    order = order[np.argsort(-sizes[order], kind="stable")]

    k = len(names)
    size_t = [f * n for f in fractions]
    lab_t = [[f * float(lab_total[j]) for j in range(n_lab)] for f in fractions]
    have = [0.0] * k
    have_lab = [[0.0] * n_lab for _ in range(k)]
    place = np.empty(n_comp, dtype=np.int64)
    use_labels = label_weight > 0 and n_lab > 0
    for c in order.tolist():
        nc = float(sizes[c])
        yc = comp_lab[c].tolist() if use_labels else []
        best, best_cost = 0, float("inf")
        for s in range(k):
            t = size_t[s]
            if t <= 0:
                continue
            cost = (2.0 * (have[s] - t) * nc + nc * nc) / t
            if use_labels:
                hl, tl = have_lab[s], lab_t[s]
                extra = 0.0
                for j in range(n_lab):
                    y = yc[j]
                    if y and tl[j] > 0:
                        extra += (2.0 * (hl[j] - tl[j]) * y + y * y) / tl[j]
                cost += label_weight * extra
            if cost < best_cost:
                best, best_cost = s, cost
        place[c] = best
        have[best] += nc
        if use_labels:
            for j in range(n_lab):
                have_lab[best][j] += yc[j]
    assign = place[comps] if n else np.empty(0, dtype=np.int64)
    return Resplit(names, assign, fractions, label_names, label_ids)
