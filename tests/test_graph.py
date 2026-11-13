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


def test_link_components_match_networkx_on_an_audited_dataset():
    from splitguard.audit import AuditConfig, build_links
    from splitguard.data import from_rows
    from splitguard.graph import ComponentSummary, link_components
    from splitguard.text import jaccard, normalize, shingles

    rng = np.random.default_rng(11)
    words = [f"v{i}" for i in range(60)]
    base = [" ".join(rng.choice(words, size=8)) for _ in range(40)]
    rows = {"train": [], "test": []}
    for name in rows:
        for _ in range(60):
            t = base[int(rng.integers(0, len(base)))].split()
            if rng.random() < 0.6:
                t[int(rng.integers(0, len(t)))] = str(rng.choice(words))
            rows[name].append({"t": " ".join(t), "g": f"g{int(rng.integers(0, 70))}"})
    ds = from_rows(list(rows.items()), ["t"], group_fields=["g"])
    cfg = AuditConfig(kinds=("exact", "near", "group"), threshold=0.75, shingle_n=3)
    labels = link_components(len(ds), build_links(ds, cfg))

    sets = [shingles(normalize(v[0]), 3) for v in ds.values]
    g = nx.Graph()
    g.add_nodes_from(range(len(ds)))
    for i in range(len(ds)):
        for j in range(i + 1, len(ds)):
            if jaccard(sets[i], sets[j]) >= 0.75 or ds.groups["g"][i] == ds.groups["g"][j]:
                g.add_edge(i, j)
    expected = {frozenset(c) for c in nx.connected_components(g)}
    assert partition(labels) == expected

    summary = ComponentSummary(labels, np.array(ds.split), ds.split_names, ds.ids)
    d = summary.to_dict()
    crossing = [c for c in expected if len({ds.split[i] for i in c}) == 2]
    assert d["crossing"]["components"] == len(crossing)
    assert d["crossing"]["largest"] == max(len(c) for c in crossing)
    assert d["components"] == len(expected)
    assert sum(d["size_histogram"].values()) == len(expected)
    top = d["largest_crossing"][0]
    assert top["size"] == sum(top["by_split"].values()) == d["crossing"]["largest"]
