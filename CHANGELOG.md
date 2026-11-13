# Changelog

## 0.1.0 (2026-11-13)

First release.

- Load two or more splits (or one file to split) from JSONL, JSON arrays, CSV, TSV and
  Parquet (with the `parquet` extra); list-valued fields are kept item by item.
- `splitguard audit`: exact duplicates after normalisation, near-duplicates (character or
  word shingles, MinHash with LSH banding, exact Jaccard check), shared group-column values
  and shared field values across splits; records affected and their share per split, linked
  pairs, label agreement of linked pairs, example ids; Markdown or JSON.
- `splitguard components`: the record graph with every link as an edge, its connected
  components, and the components that cross a split boundary.
- `splitguard resplit`: a seeded greedy assignment of whole components to new splits with
  size and label targets, reporting how far it ended up from both.
- `splitguard effect`: a hashed word n-gram logistic regression trained and scored on the
  original split (all, leaked and clean test records) and on a leak-free re-split, with
  Wilson and Newcombe intervals.
