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



## Preparing a downloaded TableShift archive on the server

If the course Box archive has already been downloaded into this project directory, first identify the archive filename:

```bash
pwd
find . -maxdepth 2 -type f \( -name "*.zip" -o -name "*.tar" -o -name "*.tar.gz" -o -name "*.tgz" -o -name "*.tar.bz2" -o -name "*.tar.xz" \) -print
```

Then run the normalizer. Replace `YOUR_ARCHIVE.zip` with the filename printed by the command above:

```bash
python scripts/prepare_tableshift_csv.py \
  --archive YOUR_ARCHIVE.zip \
  --extract-dir data/raw_tableshift \
  --output data/tableshift \
  --overwrite
```

If you have already manually extracted the archive into a directory, use `--source` instead of `--archive`:

```bash
python scripts/prepare_tableshift_csv.py \
  --source path/to/extracted_tableshift_directory \
  --output data/tableshift \
  --overwrite
```

The script copies or validates the required normalized layout:

```text
data/tableshift/<dataset>/train.csv
data/tableshift/<dataset>/validation.csv
data/tableshift/<dataset>/id_test.csv
data/tableshift/<dataset>/ood_test.csv
```

The normalizer supports both already-combined CSVs and the course archive pattern with separate feature/label files such as `*_Xtrain.csv` plus `*_ytrain.csv`. For separate files, it merges the feature columns and writes a final `label` column. It also warns if a split cannot be converted to contain one of the accepted label columns: `label`, `target`, or `y`. If the archive uses a different label-column name, rename that column before running the full experiment.

After normalization, verify the result:

```bash
find data/tableshift -maxdepth 2 -type f -name "*.csv" | sort
```

You should see 24 CSV files: four splits for each of the six datasets.

If the script reports missing splits because the archive has unusual filenames, inspect the extracted tree:

```bash
find data/raw_tableshift -maxdepth 4 -type f | sort | sed -n '1,120p'
```

Then manually copy/rename files into the normalized layout, for example:

```bash
mkdir -p data/tableshift/assistments
cp path/to/assistments_train.csv data/tableshift/assistments/train.csv
cp path/to/assistments_val.csv data/tableshift/assistments/validation.csv
cp path/to/assistments_id_test.csv data/tableshift/assistments/id_test.csv
cp path/to/assistments_ood_test.csv data/tableshift/assistments/ood_test.csv
```

## Fresh clone to final run checklist

Follow these steps on a new machine or server.

### 0. Clone the repository

Replace the URL with your own repository URL if it is different:

```bash
git clone <your-repo-url>
cd Self-Weakly-Supervised-Learning-on-Tabular-Data
```

### 1. Create a Python environment

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
```

If PyTorch is not installed by the command above or you need a CUDA-specific wheel, install PyTorch from the official selector first, then rerun `pip install -e .`.

### 2. Prepare the TableShift data

Use one of the two supported data routes:

- **Recommended for the course handout:** download the course Box data and arrange CSV files under `data/tableshift/<dataset>/`.
- **Alternative:** install the official TableShift package with `pip install -e '.[tableshift]'` and run without `--csv-dir`.

For the CSV route, the final tree must be:

```text
data/tableshift/assistments/train.csv
data/tableshift/assistments/validation.csv
data/tableshift/assistments/id_test.csv
data/tableshift/assistments/ood_test.csv
data/tableshift/nhanes_lead/train.csv
...
data/tableshift/acsunemployment/ood_test.csv
```

Each CSV must include one label column named `label`, `target`, or `y`.

### 3. Verify the environment with toy data

Run this before the full experiment:

```bash
python scripts/make_toy_csv.py
swsl-run \
  --csv-dir data/toy \
  --datasets assistments \
  --seeds 0 \
  --epochs 2 \
  --tta-steps 1 \
  --methods logreg erm masked_ssl frc_tta \
  --max-rows 200 \
  --device cuda \
  --output results/toy_metrics.csv
```

If CUDA is unavailable, change `--device cuda` to `--device cpu`.

### 4. Run a small real-data debug job

This confirms the real CSV layout before launching all six datasets:

```bash
CUDA_VISIBLE_DEVICES=0 swsl-run \
  --csv-dir data/tableshift \
  --datasets assistments \
  --seeds 0 \
  --methods logreg erm masked_ssl frc_tta \
  --epochs 5 \
  --tta-steps 3 \
  --batch-size 256 \
  --device cuda \
  --output results/debug_assistments.csv
```

Use the GPU index from `nvidia-smi`; if your RTX 5090 is not GPU 0, replace `CUDA_VISIBLE_DEVICES=0` with the correct index.

### 5. Run the full assignment experiment

After the debug job succeeds, run all required datasets and three seeds:

```bash
CUDA_VISIBLE_DEVICES=0 swsl-run \
  --csv-dir data/tableshift \
  --datasets assistments nhanes_lead brfss_diabetes acsfoodstamps physionet acsunemployment \
  --seeds 0 1 2 \
  --methods logreg rf sk_mlp erm masked_ssl frc_tta \
  --epochs 30 \
  --tta-steps 20 \
  --batch-size 512 \
  --device cuda \
  --output results/tableshift_metrics.csv
```

If you encounter CUDA out-of-memory on the RTX 5090, first retry with `--batch-size 256`. If it still fails, use the H20 by changing `CUDA_VISIBLE_DEVICES` to the H20 GPU index shown by `nvidia-smi`.

### 6. Summarize results for the report

```bash
python scripts/summarize_results.py \
  --input results/tableshift_metrics.csv \
  --output results/tableshift_summary.csv
```

Use `results/tableshift_summary.csv` to fill the mean and standard deviation cells in `reports/main.tex`.

### 7. Build the PDF report

Download the NeurIPS 2025 style zip from the assignment link, copy `neurips_2025.sty` into `reports/`, then run:

```bash
cd reports
latexmk -pdf main.tex
cd ..
```

### 8. Package submission files

Use your real student ID and name in the zip filename:

```bash
zip -r StudentID_Name.zip \
  README.md pyproject.toml requirements.txt \
  src scripts reports \
  results/tableshift_metrics.csv results/tableshift_summary.csv
```

Do not include large raw datasets in the zip unless the instructor explicitly asks for them.

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


## GPU recommendation

A GPU is helpful for the PyTorch methods (`erm`, `masked_ssl`, and `frc_tta`) but is not required for the sklearn baselines. For this assignment-scale MLP, **16 GB of VRAM is usually enough**, **24--32 GB is comfortable**, and larger memory is only needed if one-hot encoded ACS/PhysioNet features become very wide or you raise the batch size substantially.

If both an NVIDIA H20 and an RTX 5090 are available, use the **RTX 5090 first** for this project:

- RTX 5090 has 32 GB VRAM and is typically faster for a single local training job of this size.
- H20 has much larger accelerator memory and is better reserved for very large jobs, multi-user servers, or cases where the 5090 runs out of memory.
- If the 5090 reports CUDA out-of-memory, first lower `--batch-size` to 256; if that is still not enough, switch to H20.

Check available GPUs:

```bash
nvidia-smi
```

Run on the first visible GPU, usually enough if the 5090 is the only CUDA device exposed:

```bash
swsl-run \
  --csv-dir data/tableshift \
  --datasets assistments nhanes_lead brfss_diabetes acsfoodstamps physionet acsunemployment \
  --seeds 0 1 2 \
  --methods logreg rf sk_mlp erm masked_ssl frc_tta \
  --epochs 30 \
  --tta-steps 20 \
  --batch-size 512 \
  --device cuda \
  --output results/tableshift_metrics.csv
```

To force a specific GPU, choose the GPU index shown by `nvidia-smi`. For example, if the RTX 5090 is GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 swsl-run \
  --csv-dir data/tableshift \
  --datasets assistments nhanes_lead brfss_diabetes acsfoodstamps physionet acsunemployment \
  --seeds 0 1 2 \
  --methods logreg rf sk_mlp erm masked_ssl frc_tta \
  --epochs 30 \
  --tta-steps 20 \
  --batch-size 512 \
  --device cuda \
  --output results/tableshift_metrics.csv
```

If the H20 is GPU 1 and you want to run there instead, change only the device mask:

```bash
CUDA_VISIBLE_DEVICES=1 swsl-run \
  --csv-dir data/tableshift \
  --datasets assistments nhanes_lead brfss_diabetes acsfoodstamps physionet acsunemployment \
  --seeds 0 1 2 \
  --methods logreg rf sk_mlp erm masked_ssl frc_tta \
  --epochs 30 \
  --tta-steps 20 \
  --batch-size 512 \
  --device cuda \
  --output results/tableshift_metrics.csv
```

For CPU-only debugging, use fewer datasets, one seed, fewer epochs, and a smaller batch size:

```bash
swsl-run \
  --csv-dir data/tableshift \
  --datasets assistments \
  --seeds 0 \
  --methods logreg erm masked_ssl frc_tta \
  --epochs 5 \
  --tta-steps 3 \
  --batch-size 256 \
  --device cpu \
  --output results/debug_cpu_metrics.csv
```

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
