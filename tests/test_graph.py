import networkx as nx
import numpy as np

from splitguard.graph import DisjointSet, connected_components


def partition(labels: np.ndarray) -> set[frozenset[int]]:
    groups: dict[int, set[int]] = {}
    for i, c in enumerate(labels.tolist()):
        groups.setdefault(c, set()).add(i)
    return {frozenset(g) for g in groups.values()}


def test_components_match_networkx_on_random_graphs():
    rng = np.random.default_rng(5)
    for trial in range(20):
        n = int(rng.integers(1, 300))
        m = int(rng.integers(0, 2 * n))
        edges = [(int(a), int(b)) for a, b in rng.integers(0, n, size=(m, 2))]
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(edges)
        expected = {frozenset(c) for c in nx.connected_components(g)}
        assert partition(connected_components(n, edges)) == expected, trial


def test_labels_numbered_by_first_appearance_and_union_chain():
    dsu = DisjointSet(6)
    dsu.union_chain([4, 1, 5])
    assert dsu.labels().tolist() == [0, 1, 2, 3, 1, 1]
    assert dsu.size[dsu.find(5)] == 3
    assert dsu.union(1, 4) == dsu.find(5)
