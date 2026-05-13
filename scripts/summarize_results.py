from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize per-seed experiment metrics as mean±std tables.")
    parser.add_argument("--input", default="results/tableshift_metrics.csv", help="Per-seed metrics CSV produced by swsl-run.")
    parser.add_argument("--output", default="results/tableshift_summary.csv", help="Destination CSV for grouped mean/std metrics.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = ["accuracy", "balanced_accuracy", "f1", "generalization_gap"]
    df = pd.read_csv(args.input)
    summary = df.groupby(["dataset", "method", "split"], as_index=False)[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(col).rstrip("_") for col in summary.columns.to_flat_index()]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
