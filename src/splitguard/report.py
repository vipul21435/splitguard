"""Markdown rendering of the JSON-shaped reports."""

from __future__ import annotations

import json
from typing import Any


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.2f} %"


def _ids(members: dict[str, list[str]], limit: int = 4) -> str:
    parts = []
    for split, ids in members.items():
        more = f" +{len(ids) - limit}" if len(ids) > limit else ""
        parts.append(f"{split}: {', '.join(ids[:limit])}{more}")
    return "; ".join(parts)


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("`", "'").replace("\n", " ")


def audit_markdown(d: dict[str, Any]) -> str:
    lines = ["# Leak audit", ""]
    lines.append(
        "Splits: " + ", ".join(f"{s['name']} ({s['records']} records)" for s in d["splits"])
    )
    st = d["settings"]
    lines.append(
        f"Near-duplicates: {st['shingle_unit']} {st['shingle_n']}-gram shingles, "
        f"Jaccard >= {st['threshold']}, {st['num_perm']} MinHash permutations."
    )
    lines += [
        "",
        "Records of a split with a linked record in another split (kinds overlap; "
        "`any` is their union):",
        "",
        "| kind | split | linked to | records | share of split | pairs | label agreement |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for k in d["kinds"]:
        for row in k["by_split"]:
            lines.append(
                f"| {k['kind']} | {row['split']} | {row['linked_to']} | {row['records']} | "
                f"{_pct(row['share'])} | {row['pairs']} | {_pct(row['label_agreement'])} |"
            )
    for row in d["any"]["by_split"]:
        lines.append(
            f"| any | {row['split']} | {row['linked_to']} | {row['records']} | "
            f"{_pct(row['share'])} | - | - |"
        )
    for k in d["kinds"]:
        if not k["examples"]:
            continue
        lines += ["", f"## Examples: {k['kind']}", ""]
        for ex in k["examples"]:
            if "similarity" in ex:
                line = f"- Jaccard {ex['similarity']:.3f}: {_ids(ex['a'])} ~ {_ids(ex['b'])}"
                if "a_text" in ex:
                    line += f"; text `{_cell(ex['a_text'])}` ~ `{_cell(ex['b_text'])}`"
            else:
                line = f"- {_ids(ex['members'])}"
                if "key" in ex:
                    line += f"; key `{_cell(ex['key'])}`"
            lines.append(line)
    return "\n".join(lines) + "\n"


def dumps(d: dict[str, Any]) -> str:
    return json.dumps(d, indent=2, ensure_ascii=False) + "\n"


def components_markdown(d: dict[str, Any]) -> str:
    c = d["crossing"]
    lines = [
        "# Record graph",
        "",
        f"{d['records']} records, {d['components']} components ({d['singletons']} singletons), "
        f"largest component {d['largest']} records.",
        "",
        f"Components that cross a split boundary: {c['components']}, holding {c['records']} "
        f"records; the largest has {c['largest']} records.",
        "",
        "| split | records | in crossing components | share |",
        "|---|---:|---:|---:|",
    ]
    for row in c["by_split"]:
        lines.append(
            f"| {row['split']} | {row['records']} | {row['in_crossing']} | {_pct(row['share'])} |"
        )
    lines += ["", "| component size | components |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in d["size_histogram"].items()]
    if d["largest_crossing"]:
        lines += ["", "## Largest crossing components", ""]
        for comp in d["largest_crossing"]:
            split_counts = ", ".join(f"{k} {v}" for k, v in comp["by_split"].items())
            more = " ..." if comp["size"] > len(comp["ids"]) else ""
            lines.append(
                f"- {comp['size']} records ({split_counts}): {', '.join(comp['ids'])}{more}"
            )
    return "\n".join(lines) + "\n"
