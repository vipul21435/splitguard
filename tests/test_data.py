import json

import pytest

from splitguard.data import from_rows, load_splits, parse_split_spec, read_rows


def test_read_jsonl_csv_tsv_and_json_array(tmp_path):
    rows = [{"text": "a b", "label": "1"}, {"text": "c, d", "label": "0"}]
    (tmp_path / "x.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    (tmp_path / "x.json").write_text(json.dumps(rows))
    (tmp_path / "x.csv").write_text('text,label\na b,1\n"c, d",0\n')
    (tmp_path / "x.tsv").write_text("text\tlabel\na b\t1\nc, d\t0\n")
    for name in ("x.jsonl", "x.json", "x.csv", "x.tsv"):
        assert read_rows(tmp_path / name) == rows


def test_unsupported_suffix(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hello")
    with pytest.raises(ValueError, match="unsupported"):
        read_rows(p)


def test_from_rows_ids_labels_groups():
    ds = from_rows(
        [
            ("train", [{"q": "x", "y": 1, "g": "a"}, {"q": None, "y": None, "g": ""}]),
            ("test", [{"q": "z", "y": 0, "g": "a"}]),
        ],
        ["q"],
        label_field="y",
        group_fields=["g"],
    )
    assert ds.ids == ["train:0", "train:1", "test:0"]
    assert ds.split == [0, 0, 1]
    assert ds.values == [("x",), ("",), ("z",)]
    assert ds.labels == ["1", None, "0"]
    assert ds.groups == {"g": ["a", None, "a"]}
    assert ds.split_sizes() == {"train": 2, "test": 1}
    assert ds.has_labels


def test_missing_text_field_is_an_error():
    with pytest.raises(KeyError, match="missing text field"):
        from_rows([("train", [{"a": "x"}])], ["b"])


def test_duplicate_split_names_rejected():
    with pytest.raises(ValueError, match="distinct"):
        from_rows([("a", []), ("a", [])], ["t"])


def test_parse_split_spec(tmp_path):
    assert parse_split_spec("dev=some/where.csv")[0] == "dev"
    assert parse_split_spec("data/test.jsonl")[0] == "test"
    assert parse_split_spec("data/train.part.parquet")[0] == "train"


def test_load_splits_with_id_field(tmp_path):
    (tmp_path / "train.jsonl").write_text('{"id": 7, "t": "a"}\n')
    (tmp_path / "test.jsonl").write_text('{"id": 8, "t": "b"}\n')
    ds = load_splits(
        [str(tmp_path / "train.jsonl"), f"holdout={tmp_path / 'test.jsonl'}"], ["t"], id_field="id"
    )
    assert ds.split_names == ["train", "holdout"]
    assert ds.ids == ["7", "8"]


def test_subset_drops_empty_splits():
    from splitguard.data import subset

    ds = from_rows(
        [("a", [{"t": "x"}]), ("b", [{"t": "y"}, {"t": "z"}]), ("c", [{"t": "w"}])],
        ["t"],
        group_fields=[],
    )
    sub = subset(ds, [3, 1])
    assert sub.split_names == ["b", "c"]
    assert sub.split == [1, 0]
    assert sub.ids == ["c:0", "b:0"]
