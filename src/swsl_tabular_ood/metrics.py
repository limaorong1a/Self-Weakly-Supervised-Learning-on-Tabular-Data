from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    average = "binary" if len(np.unique(y_true)) <= 2 else "macro"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def summarize(rows: list[dict[str, object]]) -> str:
    import pandas as pd

    df = pd.DataFrame(rows)
    metric_cols = [c for c in ["accuracy", "balanced_accuracy", "f1", "generalization_gap"] if c in df]
    grouped = df.groupby(["dataset", "method", "split"], dropna=False)[metric_cols]
    stats = grouped.agg(["mean", "std"]).reset_index()
    return stats.to_string(index=False)
