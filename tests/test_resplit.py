import numpy as np
import pytest

from splitguard.resplit import parse_sizes, resplit


def random_components(seed: int, n_comp: int = 400):
    rng = np.random.default_rng(seed)
    sizes = rng.geometric(0.4, size=n_comp)
    comps = np.repeat(np.arange(n_comp), sizes)
    rng.shuffle(comps)
    labels = [("pos" if rng.random() < 0.3 else "neg") for _ in comps]
    return comps, labels


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_never_separates_a_component_and_hits_targets(seed):
    comps, labels = random_components(seed)
    r = resplit(comps, {"train": 0.7, "dev": 0.1, "test": 0.2}, labels, seed=seed)
    for c in np.unique(comps):
        assert len(np.unique(r.assign[comps == c])) == 1
    d = r.to_dict()
    assert d["records"] == len(comps)
    assert sum(row["records"] for row in d["splits"]) == len(comps)
    assert d["max_size_error"] < 0.02
    assert d["max_label_gap"] < 0.02


def test_label_weight_helps_label_balance():
    rng = np.random.default_rng(4)
    # components that are all one label, so placement decides the label mix
    comps = np.repeat(np.arange(300), rng.integers(1, 6, size=300))
    labels = ["a" if c % 3 == 0 else "b" for c in comps]
    with_labels = resplit(comps, {"train": 0.8, "test": 0.2}, labels, seed=0).to_dict()
    without = resplit(comps, {"train": 0.8, "test": 0.2}, labels, label_weight=0, seed=0)
    assert with_labels["max_label_gap"] <= without.to_dict()["max_label_gap"] + 1e-9
    assert with_labels["max_label_gap"] < 0.02


def test_seeded_and_counts_accepted():
    comps, labels = random_components(9)
    a = resplit(comps, {"train": 800, "test": 200}, labels, seed=3)
    b = resplit(comps, {"train": 0.8, "test": 0.2}, labels, seed=3)
    assert np.array_equal(a.assign, b.assign)
    assert a.fractions == pytest.approx([0.8, 0.2])


def test_one_giant_component_is_reported():
    comps = np.array([0] * 90 + list(range(1, 11)))
    r = resplit(comps, {"train": 0.5, "test": 0.5})
    d = r.to_dict()
    assert sorted(row["records"] for row in d["splits"]) == [10, 90]
    assert d["max_size_error"] == pytest.approx(0.8)
    assert "max_label_gap" not in d


def test_parse_sizes():
    assert parse_sizes("train=0.8, test=0.2") == {"train": 0.8, "test": 0.2}
    with pytest.raises(ValueError):
        parse_sizes("train")
    with pytest.raises(ValueError):
        parse_sizes("a=0,b=0")
    with pytest.raises(ValueError):
        resplit(np.array([0, 1]), {"a": 1.0}, ["x"])
