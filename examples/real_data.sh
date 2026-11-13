#!/usr/bin/env bash
# Fetch the two public datasets used in the README and rerun every number there.
# Usage: bash examples/real_data.sh [data_dir] [out_dir]
# Needs the parquet extra: pip install "splitguard[parquet] @ git+https://github.com/vipul21435/splitguard"
set -euo pipefail
D="${1:-data}"
OUT="${2:-runs}"
mkdir -p "$D/qqp" "$D/nq_open" "$OUT"
HF=https://huggingface.co/datasets
[ -f "$D/qqp/train.parquet" ] || curl -sSL -o "$D/qqp/train.parquet" \
  "$HF/nyu-mll/glue/resolve/main/qqp/train-00000-of-00001.parquet"
[ -f "$D/qqp/validation.parquet" ] || curl -sSL -o "$D/qqp/validation.parquet" \
  "$HF/nyu-mll/glue/resolve/main/qqp/validation-00000-of-00001.parquet"
[ -f "$D/nq_open/train.parquet" ] || curl -sSL -o "$D/nq_open/train.parquet" \
  "$HF/google-research-datasets/nq_open/resolve/main/nq_open/train-00000-of-00001.parquet"
[ -f "$D/nq_open/validation.parquet" ] || curl -sSL -o "$D/nq_open/validation.parquet" \
  "$HF/google-research-datasets/nq_open/resolve/main/nq_open/validation-00000-of-00001.parquet"

QQP=(train="$D/qqp/train.parquet" test="$D/qqp/validation.parquet"
     --text question1 question2 --id idx --label label)
NQ=(train="$D/nq_open/train.parquet" test="$D/nq_open/validation.parquet")

splitguard audit "${QQP[@]}" --ids-only --out "$OUT/qqp_audit.md"
splitguard components "${QQP[@]}" --out "$OUT/qqp_components.md"
splitguard effect "${QQP[@]}" --out "$OUT/qqp_effect.md"
splitguard audit "${NQ[@]}" --text question --kinds exact,near --ids-only \
  --out "$OUT/nq_questions.md"
splitguard audit "${NQ[@]}" --text question answer --kinds field --field-names answer \
  --ids-only --out "$OUT/nq_answers.md"
splitguard components "${NQ[@]}" --text question answer --kinds field --field-names answer \
  --out "$OUT/nq_components.md"
