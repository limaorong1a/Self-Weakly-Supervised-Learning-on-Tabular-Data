from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = ["assistments", "nhanes_lead", "brfss_diabetes", "acsfoodstamps", "physionet", "acsunemployment"]


def make_split(rng: np.random.Generator, n: int, shift: float) -> pd.DataFrame:
    age = rng.normal(45 + shift, 12, n)
    income = rng.lognormal(10 + 0.03 * shift, 0.5, n)
    group = rng.choice(["A", "B", "C"], size=n, p=[0.55 - 0.01 * shift, 0.30, 0.15 + 0.01 * shift])
    score = 0.04 * (age - 45) + 0.00002 * (income - np.median(income)) + (group == "C") * 0.7 + rng.normal(0, 1, n)
    label = (score > np.quantile(score, 0.55)).astype(int)
    return pd.DataFrame({"age": age, "income": income, "group": group, "label": label})


def main() -> None:
    out = Path("data/toy")
    for i, name in enumerate(DATASETS):
        rng = np.random.default_rng(100 + i)
        base = out / name
        base.mkdir(parents=True, exist_ok=True)
        make_split(rng, 700, 0.0).to_csv(base / "train.csv", index=False)
        make_split(rng, 250, 0.5).to_csv(base / "validation.csv", index=False)
        make_split(rng, 250, 0.5).to_csv(base / "id_test.csv", index=False)
        make_split(rng, 250, 6.0).to_csv(base / "ood_test.csv", index=False)
    print(f"Wrote toy data to {out}")


if __name__ == "__main__":
    main()
