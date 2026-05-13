from __future__ import annotations

import argparse
import csv
import re
import shutil
import tarfile
import zipfile
from pathlib import Path

REQUIRED_DATASETS = [
    "assistments",
    "nhanes_lead",
    "brfss_diabetes",
    "acsfoodstamps",
    "physionet",
    "acsunemployment",
]

SPLIT_ALIASES = {
    "train": ["train", "training"],
    "validation": ["validation", "val", "valid", "dev"],
    "id_test": ["id_test", "id-test", "idtest", "test_id", "test-id", "testid"],
    "ood_test": ["ood_test", "ood-test", "oodtest", "test_ood", "test-ood", "testood", "ood"],
}

LABEL_COLUMNS = {"label", "target", "y"}
ROLE_FEATURES = {"x", "features", "feature"}
ROLE_LABELS = {"y", "label", "labels", "target", "targets"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and normalize course-provided TableShift CSV archives into data/tableshift/<dataset>/<split>.csv."
    )
    parser.add_argument("--archive", default=None, help="Optional .zip/.tar/.tar.gz archive downloaded from the course Box.")
    parser.add_argument("--source", default=None, help="Optional already-extracted directory to scan.")
    parser.add_argument("--extract-dir", default="data/raw_tableshift", help="Where archives are extracted before scanning.")
    parser.add_argument("--output", default="data/tableshift", help="Normalized CSV output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing normalized CSV files.")
    return parser.parse_args()


def _extract_archive(archive: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    archive_stem = archive.name
    for suffix in [".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".zip", ".tar"]:
        archive_stem = archive_stem.removesuffix(suffix)
    destination = extract_dir / archive_stem
    destination.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(destination)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(destination)
    else:
        raise ValueError(f"Unsupported archive format: {archive}")
    return destination


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _tokens(text: str) -> list[str]:
    normalized = _normalize_text(text)
    return [token for token in normalized.split("_") if token]


def _compact(text: str) -> str:
    return "".join(_tokens(text))


def _tail_after_dataset(path: Path, dataset: str) -> str:
    stem = _normalize_text(path.stem)
    dataset_norm = _normalize_text(dataset)
    if stem.startswith(dataset_norm):
        stem = stem[len(dataset_norm) :].strip("_")
    return stem


def _detect_role(path: Path, dataset: str) -> str | None:
    tail_tokens = _tokens(_tail_after_dataset(path, dataset))
    if tail_tokens and tail_tokens[0] in ROLE_FEATURES:
        return "x"
    if tail_tokens and tail_tokens[0] in ROLE_LABELS:
        return "y"

    tail = _compact(_tail_after_dataset(path, dataset))
    if tail.startswith("x"):
        return "x"
    if tail.startswith("y"):
        return "y"
    return None


def _split_tail(path: Path, dataset: str) -> str:
    tail = _tail_after_dataset(path, dataset)
    role = _detect_role(path, dataset)
    if role is None:
        return _compact(tail)

    tail_tokens = _tokens(tail)
    if tail_tokens and tail_tokens[0] in (ROLE_FEATURES | ROLE_LABELS):
        return "".join(tail_tokens[1:])

    compact_tail = _compact(tail)
    return compact_tail[1:] if compact_tail.startswith(role) else compact_tail


def _matches_split(path: Path, dataset: str, split: str) -> bool:
    tail = _split_tail(path, dataset)
    aliases = {_compact(alias) for alias in SPLIT_ALIASES[split]}
    return tail in aliases


def _find_dataset_root(root: Path, dataset: str) -> Path | None:
    candidates = [p for p in root.rglob("*") if p.is_dir() and _normalize_text(p.name) == _normalize_text(dataset)]
    if candidates:
        return min(candidates, key=lambda p: len(p.parts))
    csv_candidates = [p for p in root.rglob("*.csv") if _normalize_text(dataset) in _normalize_text(str(p))]
    if csv_candidates:
        return min((p.parent for p in csv_candidates), key=lambda p: len(p.parts))
    return None


def _read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        return next(reader, [])


def _has_label_column(path: Path) -> bool:
    header = {col.strip() for col in _read_header(path)}
    return bool(header & LABEL_COLUMNS)


def _find_combined_file(dataset_root: Path, dataset: str, split: str) -> Path | None:
    candidates = [p for p in dataset_root.rglob("*.csv") if _matches_split(p, dataset, split) and _has_label_column(p)]
    if candidates:
        return min(candidates, key=lambda p: (len(p.parts), len(p.name)))
    return None


def _find_role_file(dataset_root: Path, dataset: str, split: str, role: str) -> Path | None:
    candidates = [
        p
        for p in dataset_root.rglob("*.csv")
        if _matches_split(p, dataset, split) and _detect_role(p, dataset) == role
    ]
    if candidates:
        return min(candidates, key=lambda p: (len(p.parts), len(p.name)))
    return None


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _read_label_values(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = [row for row in csv.reader(f) if row]
    if not rows:
        return []

    first_cell = rows[0][0].strip()
    has_header = first_cell in LABEL_COLUMNS or (not _looks_numeric(first_cell) and len(rows) > 1)
    data_rows = rows[1:] if has_header else rows
    return [row[0] for row in data_rows]


def _merge_feature_label_csv(feature_file: Path, label_file: Path, destination: Path) -> None:
    labels = _read_label_values(label_file)
    with feature_file.open(newline="", encoding="utf-8-sig") as f:
        feature_rows = list(csv.reader(f))
    if not feature_rows:
        raise ValueError(f"Feature CSV is empty: {feature_file}")

    header = feature_rows[0]
    rows = feature_rows[1:]
    if len(rows) != len(labels):
        raise ValueError(
            f"Feature/label row mismatch for {feature_file} and {label_file}: {len(rows)} features vs {len(labels)} labels"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([*header, "label"])
        for row, label in zip(rows, labels, strict=True):
            writer.writerow([*row, label])


def _write_split(dataset_root: Path, dataset: str, split: str, destination: Path, overwrite: bool) -> bool:
    if destination.exists() and not overwrite:
        print(f"[SKIP] {destination} already exists; use --overwrite to replace it")
        return _has_label_column(destination)

    combined_file = _find_combined_file(dataset_root, dataset, split)
    if combined_file is not None:
        shutil.copy2(combined_file, destination)
        print(f"[COPY] {combined_file} -> {destination}")
        return True

    feature_file = _find_role_file(dataset_root, dataset, split, "x")
    label_file = _find_role_file(dataset_root, dataset, split, "y")
    if feature_file is not None and label_file is not None:
        _merge_feature_label_csv(feature_file, label_file, destination)
        print(f"[MERGE] {feature_file} + {label_file} -> {destination}")
        return True

    print(
        f"[MISS] {dataset}/{split}: need either one combined CSV with label/target/y "
        f"or a matching X*/y* pair under {dataset_root}"
    )
    if feature_file is not None:
        print(f"       found feature file only: {feature_file}")
    if label_file is not None:
        print(f"       found label file only: {label_file}")
    return False


def _normalize_dataset(source_root: Path, output_root: Path, dataset: str, overwrite: bool) -> bool:
    dataset_root = _find_dataset_root(source_root, dataset)
    if dataset_root is None:
        print(f"[MISS] {dataset}: no dataset directory or CSV files found under {source_root}")
        return False

    output_dir = output_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for split in ["train", "validation", "id_test", "ood_test"]:
        destination = output_dir / f"{split}.csv"
        split_ok = _write_split(dataset_root, dataset, split, destination, overwrite)
        if split_ok and not _has_label_column(destination):
            print(f"[WARN] {destination} has no label column named one of {sorted(LABEL_COLUMNS)}")
            split_ok = False
        ok = split_ok and ok
    return ok


def main() -> None:
    args = _parse_args()
    if not args.archive and not args.source:
        raise SystemExit("Provide --archive path/to/file.zip or --source path/to/extracted_dir")

    source_roots: list[Path] = []
    if args.archive:
        archive = Path(args.archive).expanduser().resolve()
        source_roots.append(_extract_archive(archive, Path(args.extract_dir)))
    if args.source:
        source_roots.append(Path(args.source).expanduser().resolve())

    output_root = Path(args.output)
    all_ok = True
    for source_root in source_roots:
        print(f"Scanning {source_root}")
        for dataset in REQUIRED_DATASETS:
            all_ok = _normalize_dataset(source_root, output_root, dataset, args.overwrite) and all_ok

    print("\nExpected normalized layout:")
    for dataset in REQUIRED_DATASETS:
        for split in ["train", "validation", "id_test", "ood_test"]:
            print(output_root / dataset / f"{split}.csv")

    if not all_ok:
        raise SystemExit("Some datasets/splits were missing or had label-column warnings. Fix them before the full run.")


if __name__ == "__main__":
    main()
