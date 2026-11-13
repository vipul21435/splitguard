"""Audit, inspect and re-split the toy question-pair splits, using the Python API.

Run from the repository root:  python examples/toy_walkthrough.py
"""

from pathlib import Path

import numpy as np

from splitguard.audit import AuditConfig, audit, build_links
from splitguard.data import load_splits
from splitguard.graph import ComponentSummary, link_components
from splitguard.resplit import resplit

HERE = Path(__file__).resolve().parent / "toy"

ds = load_splits(
    [str(HERE / "train.jsonl"), str(HERE / "test.jsonl")],
    ["q1", "q2"],
    id_field="id",
    label_field="label",
    group_fields=["topic"],
)
cfg = AuditConfig(kinds=("exact", "near", "field"), threshold=0.7)
links = build_links(ds, cfg)
report = audit(ds, cfg, links=links)
test, train = ds.split_names.index("test"), ds.split_names.index("train")
for result in report.results:
    ids = [ds.ids[i] for i in report.leaked(test, train, result.kind)]
    print(f"{result.kind:>6}: test records linked to train: {ids}")

labels = link_components(len(ds), links)
summary = ComponentSummary(labels, np.array(ds.split), ds.split_names, ds.ids)
print("components crossing the split:", len(summary.crossing))

new = resplit(labels, {"train": 0.6, "test": 0.4}, ds.labels, seed=0)
after = ComponentSummary(labels, new.assign, new.names, ds.ids)
print("new sizes:", dict(zip(new.names, new.sizes())), "crossing after:", len(after.crossing))
