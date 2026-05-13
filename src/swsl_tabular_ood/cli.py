from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .baselines import train_logistic_regression, train_random_forest, train_sklearn_mlp
from .data import iter_datasets, load_encoded_dataset
from .methods import TTAConfig, TrainConfig, adapt_frc_tta, predict, train_masked_ssl, train_supervised
from .metrics import classification_metrics, summarize


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TableShift self/weakly-supervised OOD experiments.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Dataset names; defaults to the six required datasets.")
    parser.add_argument("--csv-dir", default=None, help="Optional directory with <dataset>/<split>.csv files.")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--methods", nargs="*", default=["logreg", "rf", "sk_mlp", "erm", "masked_ssl", "frc_tta"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tta-steps", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="results/metrics.csv")
    parser.add_argument("--max-rows", type=int, default=None, help="Debug mode: subsample each split to this many rows.")
    return parser.parse_args()


def _limit(data, max_rows: int | None):
    if max_rows is None:
        return data
    for attr in ["x_train", "y_train", "x_val", "y_val", "x_id_test", "y_id_test", "x_ood_test", "y_ood_test"]:
        value = getattr(data, attr)
        setattr(data, attr, value[:max_rows])
    return data


def main() -> None:
    args = _parse_args()
    rows: list[dict[str, object]] = []
    for dataset in iter_datasets(args.datasets):
        encoded = _limit(load_encoded_dataset(dataset, csv_dir=args.csv_dir), args.max_rows)
        for seed in args.seeds:
            train_cfg = TrainConfig(seed=seed, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)
            trained_models = {}
            if "logreg" in args.methods:
                trained_models["logreg"] = train_logistic_regression(encoded.x_train, encoded.y_train, seed)
            if "rf" in args.methods:
                trained_models["rf"] = train_random_forest(encoded.x_train, encoded.y_train, seed)
            if "sk_mlp" in args.methods:
                trained_models["sk_mlp"] = train_sklearn_mlp(encoded.x_train, encoded.y_train, seed)
            base_model = train_supervised(encoded.x_train, encoded.y_train, train_cfg) if "erm" in args.methods else None
            if base_model is not None:
                trained_models["erm"] = base_model
            if "masked_ssl" in args.methods or "frc_tta" in args.methods:
                ssl_model, _ = train_masked_ssl(encoded.x_train, encoded.y_train, train_cfg)
                trained_models["masked_ssl"] = ssl_model
            if "frc_tta" in args.methods:
                tta_cfg = TTAConfig(steps=args.tta_steps, batch_size=args.batch_size, device=args.device)
                trained_models["frc_tta"] = adapt_frc_tta(trained_models["masked_ssl"], encoded.x_train, encoded.x_ood_test, tta_cfg)
            for method in args.methods:
                model = trained_models[method]
                for split, x, y in [
                    ("id_test", encoded.x_id_test, encoded.y_id_test),
                    ("ood_test", encoded.x_ood_test, encoded.y_ood_test),
                ]:
                    if hasattr(model, "predict") and not isinstance(model, torch.nn.Module):
                        y_pred = model.predict(x)
                    else:
                        y_pred = predict(model, x, args.batch_size, args.device)
                    metrics = classification_metrics(y, y_pred)
                    rows.append({"dataset": dataset, "seed": seed, "method": method, "split": split, **metrics})
                id_acc = rows[-2]["accuracy"]
                ood_acc = rows[-1]["accuracy"]
                rows[-1]["generalization_gap"] = float(id_acc) - float(ood_acc)
                rows[-2]["generalization_gap"] = 0.0
                print(json.dumps(rows[-2], sort_keys=True))
                print(json.dumps(rows[-1], sort_keys=True))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print("\nSummary")
    print(summarize(rows))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
