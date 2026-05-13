# Self/Weakly-Supervised Learning on Tabular OOD Data

This repository implements a complete assignment scaffold for **TableShift OOD generalization**. The proposed method is **FRC-TTA: Feature-Reliability Consistent Test-Time Adaptation**, a tabular-specific self-supervised mechanism that adapts only on unlabeled OOD features.

## Method summary

FRC-TTA starts from a classifier trained with an ID supervised loss plus masked feature reconstruction. At test time it uses OOD **features only** and optimizes:

1. prediction entropy minimization;
2. consistency between the original row and a column-wise masked row;
3. a weak source feature-statistics anchor to reduce degenerate target overfitting.

By masking named feature columns rather than exchanging patches or tokens, the objective respects the non-exchangeability of tabular features.

## Repository layout

```text
src/swsl_tabular_ood/     Python package
scripts/make_toy_csv.py   Small synthetic CSV generator for smoke tests
reports/main.tex          NeurIPS-style report draft
reports/references.bib    Bibliography for the report
results/                  Experiment outputs
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

To run the official TableShift API directly, also install TableShift:

```bash
pip install -e '.[tableshift]'
```

If your course-provided Box download contains already-preprocessed CSV files, you do **not** need the TableShift package. Arrange files as:

```text
data/tableshift/<dataset>/train.csv
data/tableshift/<dataset>/validation.csv
data/tableshift/<dataset>/id_test.csv
data/tableshift/<dataset>/ood_test.csv
```

Each CSV must contain one label column named `label`, `target`, or `y`.

## How to run

### 1. Quick smoke test without TableShift data

```bash
python scripts/make_toy_csv.py
swsl-run --csv-dir data/toy --datasets assistments --seeds 0 --epochs 2 --tta-steps 1 --methods logreg erm masked_ssl frc_tta --max-rows 200 --output results/toy_metrics.csv
```

### 2. Full assignment run on all six required datasets

Using pre-exported CSV files:

```bash
swsl-run \
  --csv-dir data/tableshift \
  --datasets assistments nhanes_lead brfss_diabetes acsfoodstamps physionet acsunemployment \
  --seeds 0 1 2 \
  --methods logreg rf sk_mlp erm masked_ssl frc_tta \
  --epochs 30 \
  --tta-steps 20 \
  --batch-size 512 \
  --output results/tableshift_metrics.csv
```

Using the TableShift API instead of CSV files:

```bash
swsl-run \
  --datasets assistments nhanes_lead brfss_diabetes acsfoodstamps physionet acsunemployment \
  --seeds 0 1 2 \
  --methods logreg rf sk_mlp erm masked_ssl frc_tta \
  --epochs 30 \
  --tta-steps 20 \
  --output results/tableshift_metrics.csv
```

The output CSV contains accuracy, balanced accuracy, F1, ID/OOD split, seed, method, and generalization gap. Use only validation data for hyperparameter tuning; never inspect OOD labels except for final reporting.

## Baselines and required reporting

The default command runs five baselines plus the proposed method:

- `logreg`: balanced logistic regression;
- `rf`: class-balanced random forest;
- `sk_mlp`: sklearn MLP;
- `erm`: PyTorch MLP trained with supervised ERM;
- `masked_ssl`: ERM plus masked feature reconstruction;
- `frc_tta`: proposed unlabeled target-feature adaptation.

Report mean and standard deviation over seeds 0, 1, and 2 for Accuracy, Balanced Accuracy, F1-score, ID vs OOD performance, and generalization gap.

## Report

The draft report is in `reports/main.tex`. For final submission, download the official NeurIPS 2025 style files from the assignment link and place `neurips_2025.sty` next to `reports/main.tex`, then compile:

```bash
cd reports
latexmk -pdf main.tex
```

The current `main.tex` includes an article fallback so it can still compile if the official style is not present, but the final PDF should use the official style.
