# splitguard

Find records, near-duplicates and groups that sit on both sides of a train/test split.

A random split of records looks fair, but it is not when records share a source, an author, a
template or a near-copy. Exact-duplicate checks miss most of this, and the inflated score only
shows when the same model is scored on a split without the leak.

Status: early. What works so far:

- loading two or more splits from JSONL, CSV, TSV or Parquet (Parquet needs the `parquet` extra);
- `splitguard audit`: records of one split linked to records of another split by
  - `exact`: the same record after Unicode NFKC, case folding and punctuation removal,
  - `near`: Jaccard similarity of character (or word) shingles at or above `--threshold`, found
    with MinHash and LSH banding and then checked exactly,
  - `group:<column>`: the same value of a group column,
  - `field`: for records with several text fields, one field value shared across splits in any
    position even when the whole record differs;
- per kind: records affected and their share of the split, linked pairs, label agreement of
  linked pairs, and example ids; Markdown or JSON output.

Not yet implemented: the component view, re-splitting, the baseline classifier and the
measured effect of a leak.

## Install

```
pip install git+https://github.com/vipul21435/splitguard
```

## Quickstart

```
splitguard audit examples/toy/train.jsonl examples/toy/test.jsonl \
    --text q1 q2 --id id --label label --group topic --threshold 0.7
```

prints a table; its rows for the test split are

```
| kind | split | linked to | records | share of split | pairs | label agreement |
|---|---|---|---:|---:|---:|---:|
| exact | test | train | 1 | 20.00 % | 1 | 100.00 % |
| near | test | train | 2 | 40.00 % | 2 | 100.00 % |
| group:topic | test | train | 3 | 60.00 % | 6 | 50.00 % |
| field | test | train | 3 | 60.00 % | 4 | 75.00 % |
| any | test | train | 3 | 60.00 % | - | - |
```

## Licence

MIT.

Built by Vipul Raj Jha · https://vipul21435.github.io
