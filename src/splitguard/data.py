"""Load dataset splits from JSONL, CSV/TSV or Parquet into one table of records."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_SUFFIXES = (".jsonl", ".ndjson", ".json", ".csv", ".tsv", ".parquet")


@dataclass
class Dataset:
    """All records of all splits, in file order, split by split.

    ``values[i]`` holds the raw text of each text field of record ``i`` in the order of
    ``fields``. ``split[i]`` indexes ``split_names``. Labels and group values are kept as
    strings; ``None`` marks a missing value.
    """

    split_names: list[str]
    split: list[int]
    ids: list[str]
    fields: list[str]
    values: list[tuple[str, ...]]
    labels: list[str | None]
    groups: dict[str, list[str | None]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ids)

    def split_sizes(self) -> dict[str, int]:
        sizes = dict.fromkeys(self.split_names, 0)
        for s in self.split:
            sizes[self.split_names[s]] += 1
        return sizes

    @property
    def has_labels(self) -> bool:
        return any(label is not None for label in self.labels)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN from CSV/Parquet readers
        return ""
    return value if isinstance(value, str) else str(value)


def _as_optional(value: Any) -> str | None:
    text = _as_text(value)
    return text if text.strip() else None


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read one split file into a list of dicts, choosing the reader by suffix."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        return _read_jsonl(path)
    if suffix == ".json":
        text = path.read_text(encoding="utf-8")
        stripped = text.lstrip()
        if stripped.startswith("["):
            rows = json.loads(text)
            if not all(isinstance(r, dict) for r in rows):
                raise ValueError(f"{path}: expected a JSON array of objects")
            return rows
        return _read_jsonl(path)
    if suffix in (".csv", ".tsv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open(encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh, delimiter=delimiter))
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise ImportError(
                "reading Parquet needs pyarrow: pip install 'splitguard[parquet]'"
            ) from exc
        return pq.read_table(path).to_pylist()
    raise ValueError(f"{path}: unsupported file type {suffix!r}; use one of {SUPPORTED_SUFFIXES}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno}: expected a JSON object per line")
            rows.append(row)
    return rows


def from_rows(
    splits: Sequence[tuple[str, Iterable[dict[str, Any]]]],
    text_fields: Sequence[str],
    *,
    id_field: str | None = None,
    label_field: str | None = None,
    group_fields: Sequence[str] = (),
) -> Dataset:
    """Build a :class:`Dataset` from ``(split name, rows)`` pairs.

    Records without ``id_field`` are named ``<split>:<row index>`` (0-based).
    """
    if not text_fields:
        raise ValueError("name at least one text field")
    names = [name for name, _ in splits]
    if len(set(names)) != len(names):
        raise ValueError(f"split names must be distinct, got {names}")
    ds = Dataset(
        split_names=names,
        split=[],
        ids=[],
        fields=list(text_fields),
        values=[],
        labels=[],
        groups={g: [] for g in group_fields},
    )
    for s, (name, rows) in enumerate(splits):
        for i, row in enumerate(rows):
            missing = [f for f in text_fields if f not in row]
            if missing:
                raise KeyError(f"split {name!r}, row {i}: missing text field(s) {missing}")
            ds.split.append(s)
            if id_field is not None:
                if id_field not in row:
                    raise KeyError(f"split {name!r}, row {i}: missing id field {id_field!r}")
                ds.ids.append(_as_text(row[id_field]))
            else:
                ds.ids.append(f"{name}:{i}")
            ds.values.append(tuple(_as_text(row[f]) for f in text_fields))
            ds.labels.append(_as_optional(row.get(label_field)) if label_field else None)
            for g in group_fields:
                ds.groups[g].append(_as_optional(row.get(g)))
    return ds


def parse_split_spec(spec: str) -> tuple[str, Path]:
    """Turn ``name=path`` or ``path`` into ``(name, path)``; the name defaults to the file stem."""
    if "=" in spec:
        name, _, path = spec.partition("=")
        if name and path:
            return name, Path(path)
    path = Path(spec)
    return path.name.split(".")[0], path


def load_splits(
    specs: Sequence[str | tuple[str, str | Path]],
    text_fields: Sequence[str],
    *,
    id_field: str | None = None,
    label_field: str | None = None,
    group_fields: Sequence[str] = (),
) -> Dataset:
    """Load two or more split files (``name=path`` strings or ``(name, path)`` pairs)."""
    pairs = [parse_split_spec(s) if isinstance(s, str) else (s[0], Path(s[1])) for s in specs]
    return from_rows(
        [(name, read_rows(path)) for name, path in pairs],
        text_fields,
        id_field=id_field,
        label_field=label_field,
        group_fields=group_fields,
    )
