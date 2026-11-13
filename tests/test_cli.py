import json
from pathlib import Path

import pytest

from splitguard.cli import main

TOY = Path(__file__).resolve().parents[1] / "examples" / "toy"


def toy_args(*extra: str) -> list[str]:
    return [
        str(TOY / "train.jsonl"),
        str(TOY / "test.jsonl"),
        "--text",
        "q1",
        "q2",
        "--id",
        "id",
        "--label",
        "label",
        "--group",
        "topic",
        *extra,
    ]


def test_audit_markdown_and_json(tmp_path, capsys):
    assert main(["audit", *toy_args("--threshold", "0.7")]) == 0
    md = capsys.readouterr().out
    assert "| exact | test | train | 1 | 20.00 % |" in md
    out = tmp_path / "audit.json"
    assert main(["audit", *toy_args("--format", "json", "--out", str(out), "--ids-only")]) == 0
    d = json.loads(out.read_text())
    kinds = [k["kind"] for k in d["kinds"]]
    assert kinds == ["exact", "near", "group:topic", "field"]
    assert "key" not in d["kinds"][0]["examples"][0]
    assert d["kinds"][0]["examples"][0]["members"] == {"train": ["tr1"], "test": ["te1"]}


def test_components_command(tmp_path, capsys):
    labels = tmp_path / "labels.csv"
    args = ["components", *toy_args("--threshold", "0.7", "--write-labels", str(labels))]
    assert main(args) == 0
    md = capsys.readouterr().out
    assert "Components that cross a split boundary" in md
    rows = labels.read_text().splitlines()
    assert rows[0] == "id,split,component" and len(rows) == 14


def test_resplit_output_has_no_leaks(tmp_path, capsys):
    out_dir = tmp_path / "new"
    args = ["resplit", *toy_args("--threshold", "0.7", "--format", "json")]
    assert main([*args, "--out-dir", str(out_dir), "--seed", "2"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["crossing_before"] > 0 and d["crossing_after"] == 0
    assert (out_dir / "assignment.csv").exists()
    again = [
        "audit",
        str(out_dir / "train.jsonl"),
        str(out_dir / "test.jsonl"),
        *toy_args("--threshold", "0.7", "--format", "json")[2:],
    ]
    assert main(again) == 0
    audit_d = json.loads(capsys.readouterr().out)
    assert all(row["records"] == 0 for row in audit_d["any"]["by_split"])


def test_resplit_one_file_needs_sizes(tmp_path, capsys):
    one = [str(TOY / "train.jsonl"), "--text", "q1", "q2", "--id", "id"]
    with pytest.raises(SystemExit):
        main(["resplit", *one])
    assert main(["resplit", *one, "--sizes", "a=0.5,b=0.5", "--format", "json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert [row["split"] for row in d["splits"]] == ["a", "b"]
