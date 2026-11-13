import numpy as np
import pytest
from scipy.stats import norm

from splitguard.audit import AuditConfig
from splitguard.data import from_rows
from splitguard.effect import EffectConfig, measure_effect, newcombe, wilson


def test_wilson_hand_worked_and_edges():
    # 8 of 10: centre (0.8 + 1.92/10)/(1.384), known limits 0.4902 and 0.9433
    lo, hi = wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4)
    assert hi == pytest.approx(0.9433, abs=1e-4)
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == pytest.approx(1.0)
    assert wilson(0, 0) == (0.0, 1.0)
    z = norm.ppf(0.975)
    assert wilson(5, 20, z=z) == pytest.approx(wilson(5, 20))


def test_newcombe_hand_worked():
    # Newcombe (1998) example: 56/70 against 48/80, interval 0.0524 to 0.3339
    lo, hi = newcombe(56, 70, 48, 80)
    assert lo == pytest.approx(0.0524, abs=1e-4)
    assert hi == pytest.approx(0.3339, abs=1e-4)


def leaky_dataset(seed: int = 0):
    """Labels that depend on a hidden per-template coin, so only memorising a template helps."""
    rng = np.random.default_rng(seed)
    vocab = [f"t{i}" for i in range(3000)]
    templates = [list(rng.choice(vocab, size=12)) for _ in range(300)]
    label_of = rng.integers(0, 2, size=300)
    train, test = [], []
    for j in range(300):
        copies = 3 if j < 150 else 1
        for c in range(copies):
            words = list(templates[j])
            words[int(rng.integers(0, 12))] = str(rng.choice(vocab))
            row = {"t": " ".join(words), "y": str(label_of[j])}
            if j < 150 and c == 0:
                test.append(row)  # its siblings sit in train: leaked
            elif j >= 150 and j % 2 == 0:
                test.append(row)  # no sibling anywhere: clean
            else:
                train.append(row)
    return from_rows([("train", train), ("test", test)], ["t"], label_field="y")


def test_effect_finds_the_inflated_score():
    ds = leaky_dataset()
    d = measure_effect(
        ds,
        AuditConfig(kinds=("exact", "near"), threshold=0.6),
        EffectConfig(bits=14, C=10.0),
    )
    o = d["original"]
    assert d["test_leaked_records"] == 150
    assert o["leaked"]["n"] == 150 and o["clean"]["n"] == 75
    assert o["leaked"]["accuracy"] > 0.9
    assert 0.3 < o["clean"]["accuracy"] < 0.7
    lo, hi = o["leaked_minus_clean"]["ci95"]
    assert lo > 0.2
    assert d["resplit"]["test"]["accuracy"] < o["test"]["accuracy"]
    assert d["records"] == {"train": 375, "test": 225}
    assert d["resplit"]["train_records"] + d["resplit"]["test"]["n"] == 600


def test_effect_needs_two_splits_and_labels():
    ds = from_rows(
        [("a", [{"t": "x", "y": "1"}]), ("b", [{"t": "y", "y": "1"}])], ["t"], label_field="y"
    )
    with pytest.raises(ValueError, match="two classes"):
        measure_effect(ds)
    with pytest.raises(ValueError, match="different splits"):
        measure_effect(ds, cfg=EffectConfig(train="a", test="a"))
