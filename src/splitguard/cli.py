"""Command line: ``splitguard audit|components|resplit|effect``."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import __version__
from .audit import AuditConfig, audit, build_links
from .data import Dataset, load_splits
from .graph import ComponentSummary, link_components
from .report import audit_markdown, components_markdown, dumps


def _add_input(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "splits",
        nargs="+",
        help="split files as name=path or path (name = file stem); JSONL, CSV, TSV or Parquet",
    )
    p.add_argument("--text", nargs="+", required=True, help="text field(s) of a record")
    p.add_argument("--id", dest="id_field", help="id field (default: <split>:<row>)")
    p.add_argument("--label", dest="label_field", help="label field")
    p.add_argument("--group", nargs="*", default=[], help="group column(s)")


def _add_links(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--kinds",
        default="exact,near,group,field",
        help="comma-separated link kinds: exact, near, group, field (default: all)",
    )
    p.add_argument("--threshold", type=float, default=0.8, help="near-duplicate Jaccard")
    p.add_argument("--shingle", type=int, default=5, help="shingle length (default 5)")
    p.add_argument("--unit", choices=("char", "word"), default="char", help="shingle unit")
    p.add_argument("--num-perm", type=int, default=128, help="MinHash permutations")
    p.add_argument("--min-field-chars", type=int, default=1, help="ignore shorter field values")
    p.add_argument(
        "--field-names",
        nargs="*",
        default=[],
        help="text fields compared by the field kind (default: all, when there are two or more)",
    )
    p.add_argument("--seed", type=int, default=0)


def _add_output(p: argparse.ArgumentParser) -> None:
    p.add_argument("--format", choices=("md", "json"), default="md")
    p.add_argument("--out", help="write the report here instead of stdout")


def _config(args: argparse.Namespace) -> AuditConfig:
    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
    return AuditConfig(
        kinds=kinds,
        threshold=args.threshold,
        shingle_n=args.shingle,
        shingle_unit=args.unit,
        num_perm=args.num_perm,
        seed=args.seed,
        min_field_chars=args.min_field_chars,
        field_names=tuple(args.field_names),
        max_examples=getattr(args, "examples", 5),
        show_text=not getattr(args, "ids_only", False),
    )


def _load(args: argparse.Namespace) -> Dataset:
    return load_splits(
        args.splits,
        args.text,
        id_field=args.id_field,
        label_field=args.label_field,
        group_fields=args.group,
    )


def _emit(text: str, out: str | None) -> None:
    if out:
        Path(out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def cmd_audit(args: argparse.Namespace) -> int:
    ds = _load(args)
    if len(ds.split_names) < 2:
        raise SystemExit("audit needs two or more split files")
    d = audit(ds, _config(args)).to_dict()
    _emit(dumps(d) if args.format == "json" else audit_markdown(d), args.out)
    return 0


def summarise_components(ds: Dataset, cfg: AuditConfig) -> ComponentSummary:
    labels = link_components(len(ds), build_links(ds, cfg))
    return ComponentSummary(labels, np.array(ds.split, np.int64), list(ds.split_names), ds.ids)


def cmd_components(args: argparse.Namespace) -> int:
    ds = _load(args)
    summary = summarise_components(ds, _config(args))
    d = summary.to_dict(examples=args.examples)
    d["settings"] = _config(args).as_dict()
    _emit(dumps(d) if args.format == "json" else components_markdown(d), args.out)
    if args.write_labels:
        with open(args.write_labels, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "split", "component"])
            for i, comp in enumerate(summary.labels.tolist()):
                w.writerow([ds.ids[i], ds.split_names[ds.split[i]], comp])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="splitguard",
        description="Find what leaks between the splits of a dataset, and re-split without it.",
    )
    parser.add_argument("--version", action="version", version=f"splitguard {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("audit", help="count records linked across splits")
    _add_input(p)
    _add_links(p)
    _add_output(p)
    p.add_argument("--examples", type=int, default=5, help="examples per kind")
    p.add_argument("--ids-only", action="store_true", help="leave record text out of examples")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("components", help="connected components of the record graph")
    _add_input(p)
    _add_links(p)
    _add_output(p)
    p.add_argument("--examples", type=int, default=5, help="largest crossing components to list")
    p.add_argument("--write-labels", help="write id,split,component rows to this CSV file")
    p.set_defaults(func=cmd_components)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
