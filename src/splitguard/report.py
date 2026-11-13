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


def resplit_markdown(d: dict[str, Any]) -> str:
    lines = [
        "# Re-split",
        "",
        f"{d['records']} records in {d['components']} components (largest "
        f"{d['largest_component']} records); every component placed whole. Components that "
        f"cross the new splits: {d['crossing_after']} (before: {d['crossing_before']}).",
        "",
        "| split | target | records | fraction | size error | label gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in d["splits"]:
        gap = _pct(row.get("label_gap")) if "label_gap" in row else "-"
        lines.append(
            f"| {row['split']} | {_pct(row['target_fraction'])} | {row['records']} | "
            f"{_pct(row['fraction'])} | {100 * row['size_error']:+.2f} % | {gap} |"
        )
    if "label_mix_overall" in d:
        mix = ", ".join(f"{k} {_pct(v)}" for k, v in d["label_mix_overall"].items())
        lines += [
            "",
            f"Label mix over all records: {mix}. Label gap is the largest absolute difference "
            "between a split's label share and the overall share.",
        ]
    return "\n".join(lines) + "\n"


def _acc(s: dict[str, Any]) -> str:
    if s["accuracy"] is None:
        return f"- (n = {s['n']})"
    lo, hi = s["ci95"]
    return f"{100 * s['accuracy']:.2f} % [{100 * lo:.2f}, {100 * hi:.2f}] (n = {s['n']})"


def _dif(s: dict[str, Any]) -> str:
    if s["difference"] is None:
        return "-"
    lo, hi = s["ci95"]
    return f"{100 * s['difference']:+.2f} points [{100 * lo:+.2f}, {100 * hi:+.2f}]"


def effect_markdown(d: dict[str, Any]) -> str:
    o, r = d["original"], d["resplit"]
    lines = [
        "# Effect of the leak on one weak baseline",
        "",
        f"Model: {d['note']}. Trained on `{d['train']}` ({d['records']['train']} records), "
        f"scored on `{d['test']}` ({d['records']['test']} records), of which "
        f"{d['test_leaked_records']} ({_pct(d['test_leaked_share'])}) have a linked record "
        "in the training split.",
        "",
        "| scored on | accuracy, 95 % interval | majority-class rate |",
        "|---|---|---:|",
        f"| original test, all | {_acc(o['test'])} | {_pct(o['majority_class_rate'])} |",
        f"| original test, leaked records | {_acc(o['leaked'])} | |",
        f"| original test, clean records | {_acc(o['clean'])} | |",
        f"| re-split test (no crossing components) | {_acc(r['test'])} | "
        f"{_pct(r['majority_class_rate'])} |",
        "",
        f"Leaked minus clean: {_dif(o['leaked_minus_clean'])}.",
        f"Original minus re-split: {_dif(d['original_minus_resplit'])}.",
        "",
        "The re-split pools both splits and moves whole components, so its test set differs "
        "from the original one in more ways than the leak; read the second difference with "
        "that in mind.",
    ]
    return "\n".join(lines) + "\n"
