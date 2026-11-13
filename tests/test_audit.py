import itertools

import numpy as np
import pytest

from splitguard.audit import AuditConfig, audit
from splitguard.data import from_rows
from splitguard.text import jaccard, normalize, shingles

WORDS = [f"tok{i}" for i in range(500)]


def random_text(rng: np.random.Generator, k: int = 15) -> str:
    return " ".join(rng.choice(WORDS, size=k))


def clean_dataset(seed: int = 0, n_train: int = 80, n_test: int = 30):
    rng = np.random.default_rng(seed)
    train = [{"t": random_text(rng), "y": str(i % 2), "g": f"g{i}"} for i in range(n_train)]
    test = [{"t": random_text(rng), "y": str(i % 2), "g": f"h{i}"} for i in range(n_test)]
    return train, test


def test_no_leaks_reported_on_clean_data():
    train, test = clean_dataset()
    ds = from_rows([("train", train), ("test", test)], ["t"], label_field="y", group_fields=["g"])
    rep = audit(ds)
    for r in rep.results:
        assert r.bits.sum() == 0, r.kind
        assert r.pairs.sum() == 0
        assert r.examples == []
    assert rep.any_bits.sum() == 0
    assert all(row["records"] == 0 for row in rep.to_dict()["any"]["by_split"])


def test_planted_exact_near_and_group_leaks_are_found():
    train, test = clean_dataset(1)
    # exact copy (differs only in case and punctuation) of train 3 in test 0
    test[0]["t"] = train[3]["t"].upper() + "!!"
    # near copy of train 5 in test 1: one word changed out of 15
    words = train[5]["t"].split()
    words[7] = "zzzz"
    test[1]["t"] = " ".join(words)
    # group shared by train 9 and test 2
    test[2]["g"] = train[9]["g"]
    ds = from_rows([("train", train), ("test", test)], ["t"], label_field="y", group_fields=["g"])
    rep = audit(ds, AuditConfig(threshold=0.6))
    ex, near, grp = rep.result("exact"), rep.result("near"), rep.result("group:g")
    test_ids = lambda kind: {ds.ids[i] for i in rep.leaked(1, 0, kind)}  # noqa: E731
    assert test_ids("exact") == {"test:0"}
    assert test_ids("near") == {"test:0", "test:1"}
    assert test_ids("group:g") == {"test:2"}
    assert {ds.ids[i] for i in rep.leaked(0, 1)} == {"train:3", "train:5", "train:9"}
    assert ex.pairs[0, 1] == ex.pairs[1, 0] == 1
    assert near.pairs[0, 1] == 2
    assert grp.pairs[0, 1] == 1
    # label agreement: test 0 has label 0, train 3 has label 1
    assert ex.labelled[0, 1] == 1 and ex.agree[0, 1] == 0
    assert near.examples[0]["a"] or near.examples[0]["b"]
    d = rep.to_dict()
    row = next(r for r in d["any"]["by_split"] if r["split"] == "test")
    assert row["records"] == 3
    assert row["share"] == pytest.approx(3 / 30)


def test_shared_field_across_positions_without_whole_record_match():
    train = [{"a": "how do I learn python", "b": "best way to learn python"}]
    test = [
        {"a": "what is a good laptop", "b": "How do I learn Python?"},
        {"a": "unrelated question here", "b": "another one entirely"},
    ]
    ds = from_rows([("train", train), ("test", test)], ["a", "b"])
    rep = audit(ds, AuditConfig(kinds=("exact", "field")))
    assert rep.result("exact").bits.sum() == 0
    assert [ds.ids[i] for i in rep.leaked(1, 0, "field")] == ["test:0"]
    assert rep.result("field").examples[0]["key"] == "how do i learn python"


def test_record_counts_match_brute_force_on_random_data():
    rng = np.random.default_rng(7)
    base = [random_text(rng, 10) for _ in range(25)]
    rows = {"a": [], "b": [], "c": []}
    for name in rows:
        for _ in range(30):
            words = base[int(rng.integers(0, len(base)))].split()
            if rng.random() < 0.5:
                words[int(rng.integers(0, len(words)))] = str(rng.choice(WORDS))
            rows[name].append({"t": " ".join(words), "y": str(int(rng.integers(0, 3)))})
    ds = from_rows(list(rows.items()), ["t"], label_field="y")
    cfg = AuditConfig(kinds=("exact", "near"), threshold=0.7, shingle_n=4)
    rep = audit(ds, cfg)
    sets = [shingles(normalize(v[0]), 4) for v in ds.values]
    n = len(ds)
    for kind in ("exact", "near"):

        def linked(i, j, kind=kind):
            if kind == "exact":
                return normalize(ds.values[i][0]) == normalize(ds.values[j][0])
            return jaccard(sets[i], sets[j]) >= 0.7

        res = rep.result(kind)
        for s, t in itertools.permutations(range(3), 2):
            expect_rec = {
                j
                for j in range(n)
                if ds.split[j] == t and any(ds.split[i] == s and linked(i, j) for i in range(n))
            }
            assert set(rep.leaked(t, s, kind).tolist()) == expect_rec
            pairs = [
                (i, j)
                for i in range(n)
                for j in range(n)
                if ds.split[i] == s and ds.split[j] == t and linked(i, j)
            ]
            assert res.pairs[s, t] == len(pairs)
            assert res.agree[s, t] == sum(ds.labels[i] == ds.labels[j] for i, j in pairs)


def test_unknown_kind_rejected():
    ds = from_rows([("a", [{"t": "x"}]), ("b", [{"t": "y"}])], ["t"])
    with pytest.raises(ValueError, match="unknown"):
        audit(ds, AuditConfig(kinds=("exact", "fuzzy")))


def test_list_valued_answers_and_field_subset():
    train = [{"q": "who wrote hamlet", "a": ["Shakespeare", "William Shakespeare"]}]
    test = [
        {"q": "who wrote macbeth", "a": ["william shakespeare"]},
        {"q": "capital of peru", "a": ["Lima"]},
    ]
    ds = from_rows([("train", train), ("test", test)], ["q", "a"])
    rep = audit(ds, AuditConfig(kinds=("field",), field_names=("a",)))
    assert [ds.ids[i] for i in rep.leaked(1, 0, "field")] == ["test:0"]
    assert rep.to_dict()["settings"]["field_names"] == ["a"]
    with pytest.raises(ValueError, match="not text fields"):
        audit(ds, AuditConfig(kinds=("field",), field_names=("answer",)))
