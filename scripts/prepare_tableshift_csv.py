from __future__ import annotations

import argparse
import csv
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
    "id_test": ["id_test", "id-test", "idtest", "test_id", "test-id", "testid", "id"],
    "ood_test": ["ood_test", "ood-test", "oodtest", "test_ood", "test-ood", "testood", "ood"],
}

LABEL_COLUMNS = {"label", "target", "y"}


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


def _canonical_name(path: Path) -> str:
    return path.stem.lower().replace(" ", "_")


def _find_dataset_root(root: Path, dataset: str) -> Path | None:
    candidates = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() == dataset]
    if candidates:
        return min(candidates, key=lambda p: len(p.parts))
    csv_candidates = [p for p in root.rglob("*.csv") if dataset in str(p).lower()]
    if csv_candidates:
        return min((p.parent for p in csv_candidates), key=lambda p: len(p.parts))
    return None


def _find_split_file(dataset_root: Path, dataset: str, split: str) -> Path | None:
    aliases = SPLIT_ALIASES[split]
    csv_files = list(dataset_root.rglob("*.csv"))
    dataset_named_files = [p for p in csv_files if dataset in str(p).lower()]
    if dataset_named_files:
        csv_files = dataset_named_files
    for csv_file in csv_files:
        name = _canonical_name(csv_file)
        if name == split:
            return csv_file
    for csv_file in csv_files:
        name = _canonical_name(csv_file)
        if any(name == alias or alias in name for alias in aliases):
            return csv_file
    return None


def _read_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        return next(reader, [])


def _has_label_column(path: Path) -> bool:
    header = {col.strip() for col in _read_header(path)}
    return bool(header & LABEL_COLUMNS)


def _normalize_dataset(source_root: Path, output_root: Path, dataset: str, overwrite: bool) -> bool:
    dataset_root = _find_dataset_root(source_root, dataset)
    if dataset_root is None:
        print(f"[MISS] {dataset}: no dataset directory or CSV files found under {source_root}")
        return False

    output_dir = output_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for split in ["train", "validation", "id_test", "ood_test"]:
        source_file = _find_split_file(dataset_root, dataset, split)
        destination = output_dir / f"{split}.csv"
        if source_file is None:
            print(f"[MISS] {dataset}/{split}: no matching CSV found under {dataset_root}")
            ok = False
            continue
        if destination.exists() and not overwrite:
            print(f"[SKIP] {destination} already exists; use --overwrite to replace it")
        else:
            shutil.copy2(source_file, destination)
            print(f"[COPY] {source_file} -> {destination}")
        if not _has_label_column(destination):
            print(f"[WARN] {destination} has no label column named one of {sorted(LABEL_COLUMNS)}")
            ok = False
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
