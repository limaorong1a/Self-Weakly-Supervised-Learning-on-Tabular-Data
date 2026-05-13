from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REQUIRED_TABLESHIFT_DATASETS = [
    "assistments",
    "nhanes_lead",
    "brfss_diabetes",
    "acsfoodstamps",
    "physionet",
    "acsunemployment",
]


@dataclass
class TabularSplit:
    x: pd.DataFrame
    y: np.ndarray


@dataclass
class EncodedData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_id_test: np.ndarray
    y_id_test: np.ndarray
    x_ood_test: np.ndarray
    y_ood_test: np.ndarray
    feature_names: list[str]


def _load_tableshift_dataset(name: str) -> dict[str, TabularSplit]:
    if importlib.util.find_spec("tableshift") is None:
        raise RuntimeError(
            "TableShift is not installed. Install it with `pip install -e .[tableshift]` "
            "or pass --csv-dir containing pre-exported train/validation/test CSV files."
        )
    tableshift = importlib.import_module("tableshift")
    dset = tableshift.get_dataset(name)
    raw: dict[str, TabularSplit] = {}
    split_aliases = {
        "train": ["train"],
        "validation": ["validation", "val", "id_val"],
        "id_test": ["id_test", "test"],
        "ood_test": ["ood_test", "test_ood"],
    }
    for canonical, aliases in split_aliases.items():
        last_error: Exception | None = None
        for split in aliases:
            try:
                x, y, _, _ = dset.get_pandas(split)
                raw[canonical] = TabularSplit(x=x.reset_index(drop=True), y=np.asarray(y))
                break
            except Exception as exc:  # TableShift versions use different split names.
                last_error = exc
        if canonical not in raw:
            raise RuntimeError(f"Could not load split {canonical!r} for {name}: {last_error}")
    return raw


def _read_csv_split(path: Path) -> TabularSplit:
    df = pd.read_csv(path)
    label_candidates = ["label", "target", "y"]
    label_col = next((c for c in label_candidates if c in df.columns), None)
    if label_col is None:
        raise ValueError(f"{path} must contain one of label/target/y columns")
    return TabularSplit(x=df.drop(columns=[label_col]), y=df[label_col].to_numpy())


def _load_csv_dataset(csv_dir: Path, name: str) -> dict[str, TabularSplit]:
    base = csv_dir / name
    return {
        "train": _read_csv_split(base / "train.csv"),
        "validation": _read_csv_split(base / "validation.csv"),
        "id_test": _read_csv_split(base / "id_test.csv"),
        "ood_test": _read_csv_split(base / "ood_test.csv"),
    }


def load_raw_dataset(name: str, csv_dir: str | None = None) -> dict[str, TabularSplit]:
    if csv_dir:
        return _load_csv_dataset(Path(csv_dir), name)
    return _load_tableshift_dataset(name)


def _make_encoder(x_train: pd.DataFrame) -> ColumnTransformer:
    categorical = [c for c in x_train.columns if x_train[c].dtype == "object" or str(x_train[c].dtype).startswith("category")]
    numeric = [c for c in x_train.columns if c not in categorical]
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", onehot)]),
                categorical,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def load_encoded_dataset(name: str, csv_dir: str | None = None) -> EncodedData:
    raw = load_raw_dataset(name, csv_dir=csv_dir)
    encoder = _make_encoder(raw["train"].x)
    x_train = encoder.fit_transform(raw["train"].x).astype("float32")
    return EncodedData(
        x_train=x_train,
        y_train=raw["train"].y.astype("int64"),
        x_val=encoder.transform(raw["validation"].x).astype("float32"),
        y_val=raw["validation"].y.astype("int64"),
        x_id_test=encoder.transform(raw["id_test"].x).astype("float32"),
        y_id_test=raw["id_test"].y.astype("int64"),
        x_ood_test=encoder.transform(raw["ood_test"].x).astype("float32"),
        y_ood_test=raw["ood_test"].y.astype("int64"),
        feature_names=list(encoder.get_feature_names_out()),
    )


def iter_datasets(names: Iterable[str] | None) -> list[str]:
    return list(names) if names else REQUIRED_TABLESHIFT_DATASETS
