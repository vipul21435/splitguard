import json
from pathlib import Path

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
