import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression as SkLogReg

from splitguard.model import LogisticRegression, SparseRows, featurize, lbfgs


def toy_problem(seed: int, n: int = 300, d: int = 40, k: int = 2):
    rng = np.random.default_rng(seed)
    dense = (rng.random((n, d)) < 0.15) * rng.random((n, d))
    w = rng.normal(size=(d, k))
    y = np.argmax(dense @ w + rng.gumbel(size=(n, k)), axis=1)
    rows = [np.flatnonzero(r) for r in dense]
    indptr = np.concatenate(([0], np.cumsum([len(r) for r in rows])))
    indices = np.concatenate(rows)
    data = dense[np.repeat(np.arange(n), [len(r) for r in rows]), indices]
    return SparseRows(indptr, indices, data, d), dense, y


@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("C", [0.1, 1.0, 10.0])
def test_matches_scikit_learn(k, C):
    X, dense, y = toy_problem(k * 10 + int(C * 10), k=k)
    ours = LogisticRegression(C=C, gtol=1e-9, max_iter=2000).fit(X, y)
    ref = SkLogReg(C=C, tol=1e-10, max_iter=5000).fit(dense, y)
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(dense), atol=1e-5)
    if k == 2:
        np.testing.assert_allclose(ours.coef_, ref.coef_, atol=1e-4)
    assert np.array_equal(ours.predict(X), ref.predict(dense))


def test_sparse_products_match_dense():
    X, dense, _ = toy_problem(1)
    rng = np.random.default_rng(0)
    w, r = rng.normal(size=(X.n_cols, 3)), rng.normal(size=(X.n_rows, 2))
    np.testing.assert_allclose(X.dot(w), dense @ w, atol=1e-12)
    np.testing.assert_allclose(X.tdot(r), dense.T @ r, atol=1e-12)
    idx = np.array([5, 0, 5, 299])
    np.testing.assert_allclose(X.take(idx).toarray(), dense[idx], atol=0)


def test_lbfgs_on_a_quadratic():
    a = np.diag([1.0, 10.0, 100.0])
    b = np.array([1.0, -2.0, 3.0])
    x, it = lbfgs(lambda x: (0.5 * x @ a @ x - b @ x, a @ x - b), np.zeros(3), gtol=1e-9)
    np.testing.assert_allclose(x, np.linalg.solve(a, b), rtol=1e-8)
    assert 0 < it < 100


def test_featurize_rows_and_shared_grams():
    X = featurize([("the cat sat", "a cat sat"), ("", "")], bits=10)
    assert X.n_rows == 2 and X.n_cols == 1024
    lens = np.diff(X.indptr)
    assert lens[1] == 0
    # 3 + 2 grams per field, 3 + 2 in the second, shared {cat, sat, cat sat}
    assert lens[0] <= 5 + 5 + 3
    assert np.sum(X.data[: lens[0]] ** 2) == pytest.approx(1.0)
    no_shared = featurize([("the cat sat", "a cat sat")], bits=10, shared=False)
    assert np.diff(no_shared.indptr)[0] < lens[0]


def test_one_class_rejected():
    X, _, _ = toy_problem(0)
    with pytest.raises(ValueError):
        LogisticRegression().fit(X, np.zeros(X.n_rows, dtype=int))
