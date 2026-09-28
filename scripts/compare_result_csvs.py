#!/usr/bin/env python3
# compare two result CSVs on shared metric columns; exit non-zero if any cell differs.
# run: python scripts/compare_result_csvs.py <new.csv> <committed.csv>

from __future__ import annotations

import argparse
import sys

import pandas as pd

METRICS = ("auroc", "bedroc", "ef_0.005", "ef_0.01", "ef_0.05")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compare two result CSVs on their metric columns. Exit non-zero if any cell differs."
    )
    p.add_argument("new")
    p.add_argument("committed")
    p.add_argument("--tolerance", type=float, default=0.0,
                   help="Max allowed absolute difference. Defaults to 0: a re-score of "
                        "unchanged code and unchanged inputs is bit-identical, so anything "
                        "above 0 means something moved.")
    args = p.parse_args()

    new, old = (pd.read_csv(f) for f in (args.new, args.committed))
    for frame in (new, old):
        if "target" not in frame.columns:
            print("both files need a `target` column", file=sys.stderr)
            return 2
        frame.drop(frame.index[frame["target"].astype(str).str.lower() == "mean"], inplace=True)
    new, old = new.set_index("target"), old.set_index("target")

    missing = set(new.index) ^ set(old.index)
    if missing:
        print(f"target sets differ: {sorted(missing)[:10]}", file=sys.stderr)
        return 1

    # not a plain diff: schema grew after the DUD-E sweep (code_length, tie_break, head,
    # projection). match by target on the metric columns both files share; extra columns
    # print as a note, they are not a failure
    shared = [c for c in METRICS if c in new.columns and c in old.columns]
    if not shared:
        print("no metric columns in common", file=sys.stderr)
        return 2

    delta = (new[shared] - old.loc[new.index, shared]).abs()
    worst = delta.max().max()
    print(f"{len(new)} targets x {len(shared)} metrics, max |delta| = {worst}")
    if worst > args.tolerance:
        offenders = delta.max(axis=1).sort_values(ascending=False).head(5)
        print("worst targets:", file=sys.stderr)
        for target, value in offenders.items():
            print(f"  {target}: {value}", file=sys.stderr)
        return 1
    dropped = [c for c in old.columns if c not in new.columns]
    added = [c for c in new.columns if c not in old.columns]
    if added or dropped:
        print(f"(schema differs — added {added}, dropped {dropped}; metrics match)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
